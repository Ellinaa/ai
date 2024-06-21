import re
import spacy

class EntityRecognizer:
    def __init__(self, entity_types):
        self.entity_types = entity_types
        self.nlp = spacy.load("en_core_web_sm")

    def identify_entities(self, text):
        doc = self.nlp(text)
        entities = []
        for entity in doc.ents:
            if entity.label_ in self.entity_types:
                entities.append((entity.text, entity.label_))
        return entities

class EntityAligner:
    def align_entities(self, entities, dialogue_context):
        entity_context_mapping = {}
        for entity, entity_type in entities:
            entity_context_mapping[entity] = dialogue_context
        return entity_context_mapping

# 使用例子
entity_types = ["PERSON", "LOCATION", "DATE"]
recognizer = EntityRecognizer(entity_types)
aligner = EntityAligner()

text = "John Smith is going to New York next week."
entities = recognizer.identify_entities(text)
entity_context_mapping = aligner.align_entities(entities, "John Smith is planning a trip.")
print(entity_context_mapping)
