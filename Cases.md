# KnowCoder Dataset Cases

# Outline
- [Schema Understanding Phase](#schema-understanding-phase-pretrain)
- [Schema Following Phase](#schema-following-phase-sft)


# Schema Understanding Phase (Pretrain)

## NER (Named Entity Recognition)
```
# Extract the entities from the following sentence.
sentence = "Lalita Yauhleuskaya competed at the 2008 Summer Olympics."

from Entities import Human

results = [
	Human("Lalita Yauhleuskaya")
]
```

## RE (Relation Extraction)

```
# Extract the relations from the following sentence.
sentence = "Gzim Istrefi plays for Carlstad United BK."

from Entities import Human, AssociationFootballClub
from Relations import MemberOfSportsTeam

results = [
    MemberOfSportsTeam(
        Human("Gzim Istrefi"),
        AssociationFootballClub("Carlstad United BK")
    )
]
```

## EE (Event Extraction)

```
# Extract the events from the following sentence.
sentence = "Jamsilsaenae station is adjacent to Sports Complex station which is on the Seoul Subway Line 2. The Sports Complex station is in the direction of Inner Ring Road and is located near Gangnam station."

from Entites import Entity
from Events import AdjacentStation

results = [
    AdjacentStation(
        connecting_line=[Entity("Seoul Subway Line 2")],
        towards=[Entity("Gangnam station")],
        direction=[Entity("Inner Ring Road")]
    )
]
```

# Schema Following Phase (SFT)

## NER (Named Entity Recognition)

#### Input
```
class Entity:
    """
    The base class for all entities.
    """
    def __init__(self, name: str):
        self.name = name

class Person(Entity):
    """
    Description: being that has certain capacities or attributes constituting personhood.
    Examples: patients, Donald Trump, children, women, user, patient, Trump, President Trump, Barack Obama, people
    """
    pass

class Nationality(SocialGroup):
    """
    Description: A legal identification of a person in international law, establishing the person as a subject, a national, of a sovereign state.
    Examples: American, British, Americans, German, French, English, Japanese, Russian, Australian, Indian
    """
    pass

class TvShow(Entity):
    """
    Description:
    Examples: Game of Thrones, The Walking Dead, American Idol, Modern Family, Saturday Night Live, Doctor Who, House, The Tonight Show, Mad Men, Arrested Development
    """
    pass

"""
This is an object-oriented programming task: some Entity Classes are defined above. Please instantiate all the corresponding Entity Objects in the following sentence.
"""
sentence = "I enjoyed the series ‘Professional Master Chef’ on television and I was struck by something the judges said when commenting about two of the semi-finalists. They had been highly impressed with the dishes the chefs had presented and Michel Roux Junior remarked that, despite their very obvious skill, neither chef exhibited any arrogance or conceit. Monica Galetti replied that they didn’t need to, because their work spoke for them. "
```

#### Output
```
results = [
    TvShow("Professional Master Chef"),
    Person("Michel Roux Junior"),
    Person("Monica Galetti")
]
```


## RE (Relation Extraction)
#### Input
```
class Entity:
    """
    The base class for all entities.
    """
    def __init__(self, name: str):
        self.name = name

class Relation:
    """
    The base class for all relations.
    """
    def __init__(self, head_entity: Entity, tail_entity: Entity):
        self.head_entity = head_entity
        self.tail_entity = tail_entity

class PlaceOfBirth(Relation):
    """
    Description: Most specific known (e.g. city instead of country, or hospital instead of city) birth location of a person, animal or fictional character.
    Examples: (Australian, London), (Muhammad, Mecca), (Augustus, Rome), (Tiberius, Rome), (Mozart, Salzburg), (Charles II, London), (Sima Zhao, China), (Frederick the Great, Berlin), (Julius Caesar, Rome), (Queen Myeongui, Goryeo)
    """
    def __init__(self, head_entity: Entity, tail_entity: Entity):
        super().__init__(head_entity=head_entity, tail_entity=tail_entity)

class Population(Relation):
    """
    Description: Number of people inhabiting the place; number of people of subject.
    Examples: (civil parish, 201), (São Pedro, 201), (Machame Kusini, 13,572), (São João, 201), (unincorporated community, 15), (unincorporated community, 94), (unincorporated community, 25), (Mardekheh-ye Kuchek, 197), (Pain Halu Sara, 701), (Marenj, 1,055)
    """
    def __init__(self, head_entity: Entity, tail_entity: Entity):
        super().__init__(head_entity=head_entity, tail_entity=tail_entity)

class LocatedIn(Relation):
    """
    Description:
    Examples: (National Register of Historic Places, United States), (Ontario, Canada), (São Paulo, Brazil), (Victoria, Australia), (census-designated place, United States), (New South Wales, Australia), (California, United States), (Andes, Peru), (FAA, United States), (Norwegian, Norway)
    """
    def __init__(self, head_entity: Entity, tail_entity: Entity):
        super().__init__(head_entity=head_entity, tail_entity=tail_entity)

"""
This is an object-oriented programming task: some Relation Classes and related Entity Classes are defined above. Please instantiate all the corresponding Relation Objects in the following sentence.
"""
sentence = "Kurush is a mountain village located in the Dokuzparinsky District, in southern Dagestan. Situated at 2480–2560 m above sea level depending on the source , it is the highest continuously inhabited settlement of the Greater Caucasus and of Europe as well as the southernmost settlement in Russia. As of 2015, Kurush had a population of 813."
```

#### Output
```
results = [
    LocatedIn(Entity("Kurush"), Entity("Dokuzparinsky District")),
    LocatedIn(Entity("Dokuzparinsky District"), Entity("Dagestan")),
    Population(Entity("Kurush"), Entity("813"))
]
```

## ED (Event Detection)
#### Input
```
class Event:
    """
    The base class for all events.
    """
    def __init__(self, trigger: str, arg_names, *args):
        self.trigger = trigger
        self.arguments = {}
        for arg_name, arg_values in zip(arg_names, args):
            self.arguments[arg_name] = arg_values

class GroupMembership(Event):
    """
    Description: Organization, club or musical group to which the subject belongs.
    Examples: singer, music, musician, play, concert, performance, singing, sang, sung, sing,
    """
    def __init__(self, trigger: str, *args):
        arg_names = ["start", "role", "end", "group", "member"]
        super().__init__(trigger=trigger, arg_names=arg_names, *args)

class OlympicMedalHonor(Event):
    """
    Description: The honor associated with winning an Olympic medal.
    Examples: medal, gold, winner, win, silver, competition, bronze, victory, player, compete,
    """
    def __init__(self, trigger: str, *args):
        arg_names = ["event", "country", "medalist", "medal", "olympics"]
        super().__init__(trigger=trigger, arg_names=arg_names, *args)

class Education(Event):
    """
    Description: Educational institution attended by subject.
    Examples: school, professor, coach, graduate, student, study, master, education, pupil, lecturer,
    """
    def __init__(self, trigger: str, *args):
        arg_names = [
            "start_date",
            "degree",
            "end_date",
            "institution",
            "student",
            "specialization",
            "major_field_of_study",
        ]
        super().__init__(trigger=trigger, arg_names=arg_names, *args)

class Marriage(Event):
    """
    Description: The subject has the object as their spouse (husband, wife, partner, etc.).
    Examples: wife, married, husband, marriage, wedding, marry, couple, spouse, mistress, divorce,
    """
    def __init__(self, trigger: str, *args):
        arg_names = ["spouse", "location_of_ceremony", "type_of_union", "to", "from"]
        super().__init__(trigger=trigger, arg_names=arg_names, *args)

"""
This is an object-oriented programming task: some Event Classes are defined above. Please instantiate all the corresponding Event Objects in the following sentence.
"""
sentence = "Thomas Lincoln on June 12, 1806 married Nancy Hanks in the Richard Berry home."
```

#### Output
```
results = [
    Marriage("married")
]
```

## EAE (Event Argument Extraction)
#### Input
```
class Entity:
    """
    The base class for all entities.
    """
    def __init__(self, name: str):
        self.name = name

class Event:
    """
    The base class for all events.
    """
    def __init__(self, trigger: str):
        self.trigger = trigger

class Education(Event):
    """
    Description: Educational institution attended by subject.
    """
    def __init__(
        self,
        trigger: str,  # Examples: school, professor, coach, graduate, student, study, master, education, pupil, lecturer,
        start_date: List[Entity],
        degree: List[Entity],
        end_date: List[Entity],
        institution: List[Entity],
        student: List[Entity],
        specialization: List[Entity],
        major_field_of_study: List[Entity],
    ):
        super().__init__(trigger=trigger)
        self.start_date = start_date
        self.degree = degree
        self.end_date = end_date
        self.institution = institution
        self.student = student
        self.specialization = specialization
        self.major_field_of_study = major_field_of_study

"""
This is an object-oriented programming task: some Event Classes are defined above. Please instantiate all the corresponding Event Objects in the following sentence. It is important to note that the triggers of the events are confirmed as follows: "graduate" is the trigger of event type "Education".
"""
sentence = "Albert J. Herberger (born c. 1933) is a Vice Admiral of the United States Navy, and the first United States Merchant Marine Academy graduate to attain the rank."
```

#### Output
```
results = [
    Education(
        trigger="graduate",
        institution=[Entity("United States Merchant Marine Academy")],
        student=[Entity("Albert J. Herberger")]
    )
]
```
