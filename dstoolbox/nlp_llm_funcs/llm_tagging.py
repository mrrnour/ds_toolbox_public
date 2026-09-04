####------------------------------Data Cleaning--------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------


def parser_creator(tag_list):
    """Build a LangChain parser that constrains LLM output to a given tag vocabulary.

    Parameters
    ----------
    tag_list : list of str
        Allowed tag values; the parser rejects anything outside this set.

    Returns
    -------
    langchain.output_parsers.PydanticOutputParser
        Parser whose schema is a Pydantic model with a single ``tags``
        field typed as ``List[Literal[*tag_list]]``.
    """
    from typing import Literal

    from langchain.output_parsers import PydanticOutputParser
    from langchain_core.pydantic_v1 import BaseModel, Field

    # # # Pydantic
    class tags_list(BaseModel):
        """Pydantic schema: an LLM response carrying a list of allowed tags."""

        tags: list[Literal[tuple(tag_list)]] = Field(description="List of tags")

    parser = PydanticOutputParser(pydantic_object=tags_list)
    return parser


def chain_tagger(
    tag_list, examples, prefix, model="llama3.1:8b-instruct-q5_K_M", add_human_tag=True
):
    """Build a LangChain few-shot tagging pipeline backed by an Ollama model.

    The chain selects the most relevant in-context examples via Max Marginal
    Relevance against a Chroma vector store, formats them into a few-shot
    prompt, runs the local LLM, and parses the result through the
    Pydantic parser from :func:`parser_creator`.

    Parameters
    ----------
    tag_list : list of str
        Allowed tag vocabulary.
    examples : list of dict
        Few-shot examples; each must have keys ``Content`` and ``Tags``,
        plus ``Human_Tag`` if ``add_human_tag`` is True.
    prefix : str
        Prompt prefix (task instructions).
    model : str, optional
        Ollama model name. Defaults to ``"llama3.1:8b-instruct-q5_K_M"``.
    add_human_tag : bool, optional
        Whether the prompt input includes a pre-existing ``Human_Tag``
        column. Default True.

    Returns
    -------
    Runnable
        LangChain chain ``few_shot_prompt | llm | parser`` whose output
        is the parsed Pydantic ``tags_list`` object.
    """
    from langchain import FewShotPromptTemplate, PromptTemplate
    from langchain.llms import Ollama
    from langchain.prompts.example_selector import (
        MaxMarginalRelevanceExampleSelector,
    )
    from langchain_chroma import Chroma
    from langchain_community.embeddings import OllamaEmbeddings

    if add_human_tag:
        input_variables = ["Content", "Human_Tag"]
        suffix = "{format_instructions}\ncontent: {Content}\nHuman_Tag :{Human_Tag}\nTags:"
        examples_commaSep_tags = [
            {
                "Content": example["Content"],
                "Human_Tag": example["Human_Tag"],
                "Tags": ", ".join(example["Tags"]),
            }
            for example in examples
        ]
    else:
        input_variables = ["Content"]
        suffix = "{format_instructions}\ncontent: {Content}\nTags:"
        examples_commaSep_tags = [
            {"Content": example["Content"], "Tags": ", ".join(example["Tags"])}
            for example in examples
        ]

    example_prompt = PromptTemplate(
        # template="content: {Content}\nhuman_Tag :{Human_Tag}\ntags: {Tags}",
        template="content: {Content}\ntags: {Tags}",
        input_variables=input_variables + ["Tags"],
    )

    llm_model = Ollama(
        model=model,
        #  callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
        temperature=0,
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
    parser = parser_creator(tag_list)
    # parser = CommaSeparatedListOutputParser()

    examples_commaSep_tags = examples_commaSep_tags

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
        embeddings=OllamaEmbeddings(model=model),
        # The VectorStore class that is used to store the embeddings and do a similarity search over.
        vectorstore_cls=Chroma,  # FAISS, #,
        # The number of examples to produce.
        k=10,
        input_keys=input_variables,
    )
    few_shot_prompt_template = FewShotPromptTemplate(
        example_selector=example_selector,
        # examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=input_variables,
        # example_separator="\n\n",
        partial_variables={"format_instructions": parser.get_format_instructions()},
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
