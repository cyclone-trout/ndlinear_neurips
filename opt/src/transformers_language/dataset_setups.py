
from enum import auto, Flag

class BaseEnumOptions(Flag):
    def __str__(self):
        return self.name

    @classmethod
    def list_names(cls):
        return [m.name for m in cls]


class DatasetSetups(BaseEnumOptions):
    wikitext_2 = auto()
    wikitext_103 = auto()
    bookcorpus_and_wiki = auto()
