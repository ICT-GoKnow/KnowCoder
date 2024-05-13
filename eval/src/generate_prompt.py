import re
import json
import jinja2
from typing import List, Tuple

def extract_entity_types(class_definitions: List[str]) -> List[str]:
	"""
	从类定义的字符串列表中提取head_entity和tail_entity的类型信息。

	参数:
	- class_definitions: 包含类定义字符串的列表。

	返回值:
	- 一个列表，其中每个元素是一个元组，包含两个列表：head_entity的类型和tail_entity的类型。
	"""
	# 用于匹配类型注解的正则表达式
	head_entity_pattern = re.compile(r'head_entity:\s*([\w\s|]+)')
	tail_entity_pattern = re.compile(r'tail_entity:\s*([\w\s|]+)')

	# 存储结果的列表
	entities_types = []

	for class_def in class_definitions:
		# 查找head_entity和tail_entity的类型注解
		head_match = head_entity_pattern.search(class_def)
		tail_match = tail_entity_pattern.search(class_def)
		
		# 提取类型，如果匹配到了的话
		head_types = head_match.group(1).split('|') if head_match else []
		tail_types = tail_match.group(1).split('|') if tail_match else []

		# 清理提取的类型（移除空格）
		head_types = [ht.strip() for ht in head_types]
		tail_types = [tt.strip() for tt in tail_types]

		entities_types.extend(head_types)
		entities_types.extend(tail_types)

	return list(set(entities_types))


def generate_prompt(selected_types, sentence, ed_result):
	'''
	生成实体、关系、事件的prompt。

	参数:
	- selected_types: 一个json文件路径，文件格式如下：
		{ entity: [type1, type2, type3], relation: [type4, type5, type6], ed: [type7, type8, type9], eae: [type10, type11, type12] }
	- sentence: 一个字符串，格式如下：
		"xxx"
	- ed_result: 一个字符串，内容是模型执行eae对应的ed得到的结果, 格式如下：
		'results = []'
		'results = [\n\tLife("wedding"),\n\tJustice("hearing")\n]'

	返回值:
	- 四个字符串：实体的prompt、关系的prompt、事件ed的prompt, 事件eae的prompt。
		格式如下：
		"xxx", "xxx", "xxx", "xxx"
	'''
	#selected_types = json.loads(open(selected_types).read())
	# load the template
	template_loader = jinja2.FileSystemLoader(searchpath="./template")
	template_env = jinja2.Environment(loader=template_loader)
	entity_template = template_env.get_template("NER")
	relation_template = template_env.get_template("RE")
	ed_template = template_env.get_template("ED")
	eae_template = template_env.get_template("EAE")

	# load the mapping
	upper_mapping = json.load(open('mapping/mapping.json'))
	lower_mapping = json.load(open('mapping/inverted_mapping.json'))
	pattern = r'\b[A-Z]\w*\b'
	upper_classes = re.findall(pattern, ed_result)
	include_classes = []
	for i, class_name in enumerate(upper_classes):
		if class_name in lower_mapping:
			include_classes.extend(lower_mapping[class_name])
		else:
			include_classes.append(class_name)

	# get the class def of selected types
	entity_class_names = selected_types['entity']
	relation_class_names = selected_types['relation']
	ed_class_names = selected_types['ed']
	for i, class_name in enumerate(ed_class_names):
		if class_name in upper_mapping:
			ed_class_names[i] = upper_mapping[class_name]
	eae_class_names = selected_types['eae']
	# get the intersection of eae_class_names and include_classes
	eae_class_names = list(set(eae_class_names) & set(include_classes))

	# eae_triggers = [tuple[0] for tuple in selected_types['eae']]

	entity_class_defs = json.load(open('class_defs/entity_class_defs.json'))
	relation_class_defs = json.load(open('class_defs/relation_class_defs.json'))
	ed_class_defs = json.load(open('class_defs/ed_class_defs.json')) 
	eae_class_defs = json.load(open('class_defs/eae_class_defs.json'))

	# get the class def of selected types
	selected_entity_class_defs = [entity_class_defs[class_name] for class_name in entity_class_names]
	selected_relation_class_defs = [relation_class_defs[class_name] for class_name in relation_class_names]	
	selected_ed_class_defs = [ed_class_defs[class_name] for class_name in ed_class_names]
	selected_eae_class_defs = [eae_class_defs[class_name] for class_name in eae_class_names]

	relation_entities_types = extract_entity_types(selected_relation_class_defs)
	if 'Entity' in relation_entities_types:
		relation_entities_types.remove('Entity')
	import_entity = 'from Entities import ' + ', '.join(relation_entities_types)
	if len(relation_entities_types) == 0:
		import_entity = ''

	# trigger_prompt = [f'"{trigger}" is the trigger of event type "{type}"' for trigger, type in zip(eae_triggers, eae_class_names)]
	# trigger_prompt = '; '.join(trigger_prompt)

	# render the template
	entity_prompt = entity_template.render(sentence=sentence, class_defs=selected_entity_class_defs)
	relation_prompt = relation_template.render(sentence=sentence, class_defs=selected_relation_class_defs,
											import_entity=import_entity)
	ed_prompt = ed_template.render(sentence=sentence, class_defs=selected_ed_class_defs)
	eae_prompt = eae_template.render(sentence=sentence, class_defs=selected_eae_class_defs)
	
	# empty the prompt if no class def is selected
	if len(selected_entity_class_defs) == 0:
		entity_prompt = ""
	if len(selected_relation_class_defs) == 0:
		relation_prompt = ""
	if len(selected_ed_class_defs) == 0:
		ed_prompt = ""
	if len(selected_eae_class_defs) == 0:
		eae_prompt = ""

	return entity_prompt.strip(), relation_prompt.strip(), ed_prompt.strip(), eae_prompt.strip()

# test
#selcted_types = 'selected_types.json'
#sentence = "Hello, this is a test sentence."
#entity_prompt, relation_prompt, ed_prompt, eae_prompt = generate_prompt(selcted_types, sentence, 'results = [\n\tLife("wedding"),\n\tJustice("hearing")\n]')
#print(entity_prompt, end = '\n\n')
#print(relation_prompt, end = '\n\n')
#print(ed_prompt, end = '\n\n')
#print(eae_prompt, end = '\n\n')
#
#entity_prompt, relation_prompt, ed_prompt, eae_prompt = generate_prompt(selcted_types, sentence, 'results = [\n\tLife("wedding")')
#print(entity_prompt, end = '\n\n')
#print(relation_prompt, end = '\n\n')
#print(ed_prompt, end = '\n\n')
#print(eae_prompt, end = '\n\n')
#
#entity_prompt, relation_prompt, ed_prompt, eae_prompt = generate_prompt(selcted_types, sentence, 'results = []')
#print(eae_prompt, end = '\n\n')