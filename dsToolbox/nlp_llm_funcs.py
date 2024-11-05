####------------------------------Data Cleaning--------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
def explain_acronyms(df_text, df_acronyms):
    from tqdm import tqdm

    """
    Adds acronyms and their descriptions to the message content.

    This function reads acronyms and their full forms from an Excel file, replaces acronyms
    in the message content with their full forms, and adds short descriptions for acronyms.
k
    Parameters:
    df_text (pd.DataFrame): DataFrame containing the messages to be processed.
    df_acronyms (pd.DataFrame): DataFrame containing acronyms and their descriptions

    Returns:
    pd.DataFrame: DataFrame with updated message content.
    """
    
    print("Adding acronyms...")

    # Replace acronyms with their full forms
    ##TODO: using "^| " takes for ever???
    for start_str, end_str in ([" ", " "],["^", " "],[" ", "$"],[" ", "."],["^", "."]):
        mapDict2 = dict(zip(start_str + df_acronyms['ACRONYM'] + end_str,
                            " " + df_acronyms['STANDS FOR'] + ' (' + df_acronyms['ACRONYM'] + ') '))
        df_text['Content'] = df_text['Content'].replace(mapDict2, regex=True)

    print("Adding short description for acronyms...")
    # idx=(~df_acronyms['SHORT DESCRIPTION'].isnull())
    # mapDict =  dict(zip(df_acronyms.loc[idx, 'ACRONYM'], df_acronyms.loc[idx,'SHORT DESCRIPTION']))
    ##TODO: use apply instead of iterrows
    df_acr__descrip = df_acronyms.loc[~df_acronyms['SHORT DESCRIPTION'].isnull(), ['ACRONYM', 'SHORT DESCRIPTION']]
    for _, vals in tqdm(df_acr__descrip.iterrows(), total=df_acr__descrip.shape[0]):
        short_description, acronym = vals['SHORT DESCRIPTION'], vals['ACRONYM']
        idx = df_text['Content'].str.contains(f'({acronym})', regex=False)
        df_text.loc[idx, 'Content'] += f'\n <<Short description about {acronym}: {short_description}>> '

    return df_text

def anonymizer_text(text_to_anonymize, add_title=True, add_pronoun=True):
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig

    analyzer = AnalyzerEngine()
    entities = analyzer.get_supported_entities()
    entities = [ent for ent in entities if ent not in ["URL",
                                                       "DATE_TIME"
                                                       ]
                ]

    if add_title:
        titles_list = [
                        "Sir",
                        "Ma'am",
                        "Madam",
                        "Mr.",
                        "Mrs.",
                        "Ms.",
                        "Miss",
                        "Dr.",
                        "Professor",
                    ]
        titles_recognizer = PatternRecognizer(supported_entity="TITLE",
                                            deny_list=titles_list
                                            )
        analyzer.registry.add_recognizer(titles_recognizer)
        entities=entities+['TITLE']

    if add_pronoun:
        pronoun_recognizer = PatternRecognizer(supported_entity="PRONOUN",
                                            deny_list=["he", "He", "his", "His", "she", "She", "hers", "Hers"])
        analyzer.registry.add_recognizer(pronoun_recognizer)
        entities=entities+['PRONOUN']

    analyzer_results = analyzer.analyze(text=text_to_anonymize,
                                         entities=entities,
                                        language='en')

    anonymizer = AnonymizerEngine()
    anonymized_results = anonymizer.anonymize(
                                            text=text_to_anonymize,
                                            analyzer_results=analyzer_results,    
                                            # operators={"DEFAULT":      OperatorConfig("replace", {"new_value": "<ANONYMIZED>"}), 
                                            #         "PHONE_NUMBER": OperatorConfig("mask",    {"type":"mask", "masking_char" : "*", "chars_to_mask" : 12, "from_end" : True}),
                                            #         "TITLE":        OperatorConfig("redact",  {})
                                            #         }
                                        )

    # print(f"text: {anonymized_results.text}")
    # print("detailed response:")
    # pprint(json.loads(anonymized_results.to_json()))
    return anonymized_results.text

####------------------------------NLP--------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------

class textSimilarity:

    def __init__(self, text1, text2, method):
        self.text1 = text1
        self.text2 = text2
        self.method = method

    def similarity_word2vec(self):
        import spacy
        nlp = spacy.load('en_core_web_md')  # load a model with word vectors
        doc1 = nlp(self.text1)
        doc2 = nlp(self.text2)
        return doc1.similarity(doc2)

    def similarity_wordDist(self):
        from difflib import SequenceMatcher
        return SequenceMatcher(None, self.text1, self.text2).ratio()

    def similarity_sbert(self):
        import os
        os.environ['CURL_CA_BUNDLE'] = ''
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        #Compute embedding for both lists
        embedding_1= model.encode(self.text1, convert_to_tensor=False)
        embedding_2 = model.encode(self.text2, convert_to_tensor=False)

        return util.pytorch_cos_sim(embedding_1, embedding_2)

    def calculate_similarity(self):
        method = getattr(self, self.method, lambda: "Invalid method")
        return method()

def calculate_similarity(df, column, similarity_method):
    from itertools import combinations
    unique_values = df[column].unique()
    similarity_scores = []
    for value1, value2 in combinations(unique_values, 2):
        similarity_score = textSimilarity(value1, value2, similarity_method).calculate_similarity()
        similarity_scores.append([value1, value2, similarity_score])
    return pd.DataFrame(similarity_scores, columns=[column+'_1', column+'_2', 'similarity_score'])

def unify_similar_strings(df0, column, similarity_method='similarity_wordDist', threshold=.8):
    df=df0.copy()
    df[column] = df[column].str.strip().astype(str)
    unique_values =sorted(df[column].unique(), key=len, reverse=False)
    replaced_dict = {}
    i = 0
    while i < len(unique_values):
        j = i + 1
        while j < len(unique_values):
            similarity = textSimilarity(unique_values[i], unique_values[j], similarity_method).calculate_similarity()
            if (similarity >= threshold) & (unique_values[i] != unique_values[j]):
                replaced_dict[unique_values[i]] = replaced_dict.get(unique_values[i], []) + [(unique_values[j], (df[column] == unique_values[j]).sum())]
                df.loc[df[column] == unique_values[j], column] = unique_values[i]
                unique_values = sorted(df[column].unique(), key=len, reverse=False)  # update unique_values
                break  # break the inner loop when we find a similar string
            j += 1
        i += 1

    replaced_df = pd.DataFrame([(k, v[0], v[1]) for k, values in replaced_dict.items() for v in values], columns=['Replaced', 'Original', 'Frequency'])

    return df , replaced_df

####------------------------------Lang Chain--------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------

def parser_creator(tag_list):   
    from langchain.output_parsers import PydanticOutputParser
    from langchain_core.pydantic_v1 import BaseModel, Field
    from typing import List, Literal
    # # # Pydantic
    class tags_list(BaseModel):
        tags: List[Literal[tuple(tag_list)]] = Field(description="List of tags")

    parser = PydanticOutputParser(pydantic_object=tags_list)
    return parser

def chain_tagger(tag_list, examples, prefix, model="llama3.1:8b-instruct-q5_K_M", add_human_tag=True):
    from langchain import PromptTemplate, FewShotPromptTemplate, LLMChain
    from langchain.llms import Ollama
    from langchain.llms import HuggingFaceHub	
    from langchain import HuggingFacePipeline

    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 
    from langchain.prompts.example_selector import LengthBasedExampleSelector,SemanticSimilarityExampleSelector, MaxMarginalRelevanceExampleSelector

    from langchain.output_parsers import CommaSeparatedListOutputParser
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_chroma import Chroma
    from langchain_community.vectorstores import FAISS

    if add_human_tag:
        input_variables=["Content", "Human_Tag"]
        suffix="{format_instructions}\ncontent: {Content}\nHuman_Tag :{Human_Tag}\nTags:"
        examples_commaSep_tags=[{'Content':  example['Content'],
                                'Human_Tag': example['Human_Tag'],
                                'Tags'     : ", ".join((example['Tags']))
                                }  for example in examples]                             
    else:
        input_variables=["Content"]
        suffix="{format_instructions}\ncontent: {Content}\nTags:"
        examples_commaSep_tags=[{'Content':  example['Content'],
                                'Tags'     : ", ".join((example['Tags']))
                                }  for example in examples]

    example_prompt = PromptTemplate(
                            # template="content: {Content}\nhuman_Tag :{Human_Tag}\ntags: {Tags}",
                            template="content: {Content}\ntags: {Tags}",
                            input_variables=input_variables+['Tags']
                        )

    llm_model = Ollama(model=model, 
                        #  callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
                        temperature=0
                        )

    # llm_model = HuggingFacePipeline.from_model_id(model_id=model, 
    #                                               task="summarization", 
    #                                               model_kwargs={"temperature":0,
    #                                                              "max_length":512,
    #                                                             'do_sample':True,
    #                                                              }
    #                                             )

    # llm_model = HuggingFaceHub(
	# 	repo_id="google/flan-t5-base", 
	# 	# repo_id="google/flan-t5-large", # Gives very different outputs.
	# 	model_kwargs={"temperature":0}
	# )                    
    parser=parser_creator(tag_list)
    # parser = CommaSeparatedListOutputParser()

    examples_commaSep_tags=examples_commaSep_tags

    ### NOTE: LengthBasedExampleSelector strange behaviour, not sure why?????????:
    ###works with parser_creator                  and examples_commaSep_tags
    ###works with CommaSeparatedListOutputParser  and examples=examples 
    ### commasperated doesnot force output be only provided list

    # example_selector = LengthBasedExampleSelector(
    #                                                 examples=examples_commaSep_tags,
    #                                                 example_prompt=example_prompt,
    #                                                 max_length=2000,
    #                                             )

    ### NOTE: in SemanticSimilarityExampleSelector, tags should be a comma separated string not a list. so parser_creator can't be used which provides a list of strings so not sure why it is working?????????:
    # example_selector = SemanticSimilarityExampleSelector.from_examples(
    #                                                                     # The list of examples available to select from.
    #                                                                     examples_commaSep_tags,
    #                                                                     # The embedding class used to produce embeddings which are used to measure semantic similarity.
    #                                                                     embeddings =OllamaEmbeddings(model=model),
    #                                                                     # The VectorStore class that is used to store the embeddings and do a similarity search over.
    #                                                                     vectorstore_cls=Chroma,
    #                                                                     # The number of examples to produce.
    #                                                                     k=4,
    #                                                                     # input_keys=["Content", 'Human_Tag'],
    #                                                                 )
    example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
                                                                        # The list of examples available to select from.
                                                                        examples_commaSep_tags,
                                                                        # The embedding class used to produce embeddings which are used to measure semantic similarity.
                                                                        embeddings =OllamaEmbeddings(model=model),
                                                                        # The VectorStore class that is used to store the embeddings and do a similarity search over.
                                                                        vectorstore_cls=Chroma, #FAISS, #,
                                                                        # The number of examples to produce.
                                                                        k=10,
                                                                        input_keys=input_variables
                                                                    )
    few_shot_prompt_template = FewShotPromptTemplate(
                                                    example_selector=example_selector,
                                                    # examples=examples,
                                                    example_prompt=example_prompt,
                                                    prefix=prefix,
                                                    suffix=suffix,
                                                    input_variables=input_variables,
                                                    # example_separator="\n\n",
                                                    partial_variables={"format_instructions": parser.get_format_instructions()}
                                                )
   
    ###old version:
    # chain = LLMChain(llm=llm_model, prompt=few_shot_prompt_template)
    # response = chain.run({"content": user_content})
    # parser.parse(response)

    ###new version:
    chain = few_shot_prompt_template | llm_model | parser

    # chain_components= {'few_shot_prompt_template': few_shot_prompt_template,
    #     'llm_model': llm_model,
    #     'parser': parser
    #     }
    return chain
