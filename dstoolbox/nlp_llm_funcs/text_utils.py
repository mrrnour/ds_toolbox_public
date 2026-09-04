####------------------------------Data Cleaning--------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
def explain_acronyms(df_text, df_acronyms):
    """Expand acronyms in a message column and append short descriptions.

    Replaces each acronym with ``"<full form> (<ACRONYM>)"`` in the
    ``Content`` column of ``df_text``, then appends a short description
    for any acronym that has one.

    Parameters
    ----------
    df_text : pandas.DataFrame
        Must contain a ``Content`` column of message strings; modified
        in place and returned.
    df_acronyms : pandas.DataFrame
        Lookup with columns ``ACRONYM``, ``STANDS FOR``, and optional
        ``SHORT DESCRIPTION``.

    Returns
    -------
    pandas.DataFrame
        ``df_text`` with the ``Content`` column rewritten.
    """
    from tqdm import tqdm

    print("Adding acronyms...")

    # Replace acronyms with their full forms
    ##TODO: using "^| " takes for ever???
    for start_str, end_str in ([" ", " "], ["^", " "], [" ", "$"], [" ", "."], ["^", "."]):
        mapDict2 = dict(
            zip(
                start_str + df_acronyms["ACRONYM"] + end_str,
                " " + df_acronyms["STANDS FOR"] + " (" + df_acronyms["ACRONYM"] + ") ",
                strict=False,
            )
        )
        df_text["Content"] = df_text["Content"].replace(mapDict2, regex=True)

    print("Adding short description for acronyms...")
    # idx=(~df_acronyms['SHORT DESCRIPTION'].isnull())
    # mapDict =  dict(zip(df_acronyms.loc[idx, 'ACRONYM'], df_acronyms.loc[idx,'SHORT DESCRIPTION']))
    ##TODO: use apply instead of iterrows
    df_acr__descrip = df_acronyms.loc[
        ~df_acronyms["SHORT DESCRIPTION"].isnull(), ["ACRONYM", "SHORT DESCRIPTION"]
    ]
    for _, vals in tqdm(df_acr__descrip.iterrows(), total=df_acr__descrip.shape[0]):
        short_description, acronym = vals["SHORT DESCRIPTION"], vals["ACRONYM"]
        idx = df_text["Content"].str.contains(f"({acronym})", regex=False)
        df_text.loc[idx, "Content"] += (
            f"\n <<Short description about {acronym}: {short_description}>> "
        )

    return df_text


def anonymizer_text(text_to_anonymize, add_title=True, add_pronoun=True):
    """Redact PII (and optionally titles/pronouns) from text using Presidio.

    Parameters
    ----------
    text_to_anonymize : str
        Free-text input to scan and anonymize.
    add_title : bool, optional
        If True (default), add a custom recognizer for honorifics
        (Mr., Dr., Professor, …) so they are redacted as ``TITLE``.
    add_pronoun : bool, optional
        If True (default), redact gendered pronouns (he/she/his/her).

    Returns
    -------
    str
        The anonymized text. URL and DATE_TIME entities are deliberately
        preserved.
    """
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine

    analyzer = AnalyzerEngine()
    entities = analyzer.get_supported_entities()
    entities = [ent for ent in entities if ent not in ["URL", "DATE_TIME"]]

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
        titles_recognizer = PatternRecognizer(supported_entity="TITLE", deny_list=titles_list)
        analyzer.registry.add_recognizer(titles_recognizer)
        entities = entities + ["TITLE"]

    if add_pronoun:
        pronoun_recognizer = PatternRecognizer(
            supported_entity="PRONOUN",
            deny_list=["he", "He", "his", "His", "she", "She", "hers", "Hers"],
        )
        analyzer.registry.add_recognizer(pronoun_recognizer)
        entities = entities + ["PRONOUN"]

    analyzer_results = analyzer.analyze(text=text_to_anonymize, entities=entities, language="en")

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
