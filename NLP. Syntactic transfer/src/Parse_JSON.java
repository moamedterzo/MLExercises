import java.io.*;
import java.util.ArrayList;
import org.json.JSONArray;
import org.json.JSONObject;
import simplenlg.features.Feature;
import simplenlg.features.Tense;
import simplenlg.framework.NLGElement;
import simplenlg.framework.NLGFactory;
import simplenlg.lexicon.Lexicon;
import simplenlg.phrasespec.NPPhraseSpec;
import simplenlg.phrasespec.PPPhraseSpec;
import simplenlg.phrasespec.SPhraseSpec;
import simplenlg.phrasespec.VPPhraseSpec;
import simplenlg.realiser.english.Realiser;

public class Parse_JSON {

    static NLGFactory nlgFactory;
    static DictionaryManager dictionaryManager;

    //This feature is used to indicate whether the agent for the verb is a person
    static final String FEATURE_AGENT = "Agent";

    public static void main(String args[]) {

        try {

            //Initialize SimpleNLG objects for english
            Lexicon lexicon = Lexicon.getDefaultLexicon();
            nlgFactory = new NLGFactory(lexicon);
            Realiser realiser = new Realiser(lexicon);

            //initialize Dictionary for translation
            dictionaryManager = new DictionaryManager("Dictionary.txt");

            //Call Python code in order to get the parsed sentences
            String[] json_parsed_sentences = getParsedTreesFromPython();

            //For each sentence do a translation and print it
            for (String sentence : json_parsed_sentences) {
                String englishSentence = translateSentence(realiser, new JSONArray(sentence));
                System.out.println(englishSentence);
            }
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
        }

        System.exit(0);
    }

    //This method returns the parsed sentences, calling a Python code
    static String[] getParsedTreesFromPython() throws IOException{
        //Create Python process
        Process pythonProcess = Runtime.getRuntime().exec("python Parser.py");

        //Set reader for Python output
        BufferedReader pythonResult = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream()));
        BufferedReader pythonErrors = new BufferedReader(new InputStreamReader(pythonProcess.getErrorStream()));

        // print errors, if any
        String s;
        while ((s = pythonErrors.readLine()) != null) {
            System.out.println(s);
        }

        //return json data for parsed sentences
        return pythonResult.readLine().split("\"SENTENCE_SEPARATOR\"");
    }

    //This method translates an italian sentence parsed as a json tree into a english sentence
    static String translateSentence(Realiser realiser, JSONArray jsonArray) throws Exception{

        ArrayList<NLGElement> mainClauseElements = new ArrayList<>();
        boolean flagPassiveClause = false;

        for (Object element : jsonArray) {
            //getting the NLG element
            JSONObject jsonElement = (JSONObject)element;
            NLGElement result = getNLGElement(jsonElement);

            //adding it to the list
            mainClauseElements.add(result);

            //We manage the verb kind (active/passive)
            String name = jsonElement.names().getString(0);
            if (name.equals("VPP"))
                flagPassiveClause = true;
        }

        //Create main clause
        SPhraseSpec mainClause = nlgFactory.createClause();

        //Set verb kind (attive/passive)
        mainClause.setFeature(Feature.PASSIVE, flagPassiveClause);

        //manage clause parts
        boolean firstElement = true;
        for (NLGElement element : mainClauseElements){

            if(element.getClass() == NPPhraseSpec.class) {

                //Manage the noun element
                if (flagPassiveClause) {
                    mainClause.setObject(element);
                } else if (firstElement) {
                    mainClause.setSubject(element);
                } else {
                    mainClause.setObject(element);
                }
            }
            else if(element.getClass() == VPPhraseSpec.class){

                //Manage the verb element
                mainClause.setVerb(element);

                //set verb tense
                mainClause.setFeature(Feature.TENSE, element.getFeature(Feature.TENSE));

                if(firstElement) {
                    //If the subject is not specified, we must determine whether the subject is an agent
                    boolean flagAgent = element.getFeatureAsBoolean(FEATURE_AGENT);

                    if(flagAgent)
                        mainClause.setSubject("he");
                    else
                        mainClause.setSubject("it");
                }
            }

            firstElement = false;
        }

        //realize english sentence
        return realiser.realiseSentence(mainClause);
    }

    //It returns an NLG element from a JsonObject
    //This method is recursive
    static NLGElement getNLGElement(JSONObject jsonObject){

        //Get the type of the node
        String name = jsonObject.names().getString(0);

        switch (name){

            case "NP":

                //Manage NP node
                NPPhraseSpec resultNP = nlgFactory.createNounPhrase();

                //It indicates when we find the first noun in the sub elements
                boolean firstNoun = true;

                //Iterate for each sub element
                for(Object subElement : (JSONArray)jsonObject.get(name)){

                    //Getting subelement properties
                    JSONObject subJsonObject = (JSONObject)subElement;
                    String subJsonObjectName = subJsonObject.names().getString(0);

                    //Consider the sub node type
                    switch (subJsonObjectName){

                        case "DET":
                            //determiner
                            resultNP.setDeterminer(dictionaryManager.translateTerm(subJsonObject, subJsonObjectName).englishTerm);
                            break;

                        case "N":

                            //simple noun
                            DictionaryManager.DictionaryEntry entryNoun = dictionaryManager.translateTerm(subJsonObject, subJsonObjectName);

                            if(firstNoun){

                                //The first noun is the main one
                                resultNP.setNoun(entryNoun.englishTerm);
                                resultNP.setPlural(entryNoun.isPlural);

                                firstNoun =false;
                            }
                            else {
                                //the non-first nouns are considered as modifiers
                                resultNP.addPreModifier(entryNoun.englishTerm);
                            }

                            break;

                        case "NP":

                            //complex noun
                            if(firstNoun){
                                resultNP.setNoun(getNLGElement(subJsonObject));
                                firstNoun = false;
                            }
                            else {
                                resultNP.addPreModifier(getNLGElement(subJsonObject));
                            }

                            break;

                        case "ADJ":
                            //adjective
                            resultNP.addModifier(dictionaryManager.translateTerm(subJsonObject, subJsonObjectName).englishTerm);
                            break;

                        case "POS":
                            //possessive
                            resultNP.setSpecifier(dictionaryManager.translateTerm(subJsonObject, subJsonObjectName).englishTerm);
                            break;

                        case "PP":
                            //preprosition phrase
                            resultNP.addModifier(getNLGElement(subJsonObject));
                            break;
                    }
                }

                return  resultNP;

            case "PP":

                //Manage preposition phrase
                PPPhraseSpec resultPP = nlgFactory.createPrepositionPhrase();

                //Iterate for each sub element
                for(Object subElement : (JSONArray)jsonObject.get(name)){

                    //Get subelement data
                    JSONObject subJsonObject = (JSONObject)subElement;
                    String subJsonObjectName = subJsonObject.names().getString(0);

                    //Consider the sub element type
                    switch (subJsonObjectName){

                        case "P":
                            //preposition
                            resultPP.setPreposition(dictionaryManager.translateTerm(subJsonObject, subJsonObjectName).englishTerm);
                            break;

                        case "NP":
                            //noun phrase
                            resultPP.setObject(getNLGElement(subJsonObject));
                            break;
                    }
                }

                return  resultPP;

            case "VP":
                //Manage active verb phrase
                VPPhraseSpec resultVP = nlgFactory.createVerbPhrase();

                //Iterate for each sub element
                for(Object subElement : (JSONArray)jsonObject.get(name)) {

                    //Get sub element data
                    JSONObject subJsonObject = (JSONObject)subElement;
                    String subJsonObjectName = subJsonObject.names().getString(0);

                    //Consider the sub element type
                    switch (subJsonObjectName) {

                        case "V":
                            //verb
                            NLGElement verbElement = getNLGElement(subJsonObject);
                            resultVP.setVerb(verbElement);

                            //this feature will be used for the subject determination (if not explicit)
                            resultVP.setFeature(FEATURE_AGENT, verbElement.getFeature(FEATURE_AGENT));
                            break;

                        case "AUX":
                            //If we have the auxiliary, the verb tense is the past
                            resultVP.setFeature(Feature.TENSE, Tense.PAST);
                            break;
                    }
                }

                return resultVP;

            case "VPP":

                //Manage passive verb phrase
                //VPP has the construction of type: AUX + V + V
                VPPhraseSpec resultVPP = nlgFactory.createVerbPhrase();

                boolean firstVerb = true;

                //Iterate for each sub element
                for(Object subElement : (JSONArray)jsonObject.get(name)) {

                    //Get sub element data
                    JSONObject subJsonObject = (JSONObject) subElement;
                    String subJsonObjectName = subJsonObject.names().getString(0);

                    //Consider the sub element type
                    switch (subJsonObjectName) {

                        case "V":
                            //The first verb in a VPP phrase is discarded
                            if (firstVerb)
                                firstVerb = false;
                            else {
                                //main verb
                                resultVPP.setVerb(getNLGElement(subJsonObject));

                                //past as default for the passive verb
                                resultVPP.setFeature(Feature.TENSE, Tense.PAST);
                            }
                            break;

                        case "AUX":
                            //auxiliary, do nothing
                            break;
                    }
                }

                return resultVPP;

            case "V":
                //Manage verb
                VPPhraseSpec resultVerb = nlgFactory.createVerbPhrase();

                //Check whether the verb node contains a word or a subtree
                if(jsonObject.get(name).getClass() == String.class){
                    //Word, get translated term
                    resultVerb.setVerb(dictionaryManager.translateTerm(jsonObject, name).englishTerm);

                    //this feature will be used for the subject determination
                    resultVerb.setFeature(FEATURE_AGENT, dictionaryManager.flagAgentVerb(jsonObject, name));
                }
                else {
                    //subtree, iterate for each element
                    for(Object subElement : (JSONArray)jsonObject.get(name)) {

                        //Get sub element data
                        JSONObject subJsonObject = (JSONObject) subElement;
                        String subJsonObjectName = subJsonObject.names().getString(0);

                        //Consider sub element type
                        switch (subJsonObjectName) {

                            case "V":
                                //verb
                                //get element and set it as the verb for the main element
                                NLGElement resultSubVerb = getNLGElement(subJsonObject);
                                resultVerb.setVerb(resultSubVerb);

                                //this feature will be used for the subject determination
                                resultVerb.setFeature(FEATURE_AGENT, resultSubVerb.getFeature(FEATURE_AGENT));
                                break;

                            case "V-MOD":
                                //verb modifier
                                resultVerb.addPostModifier(dictionaryManager.translateTerm(subJsonObject, subJsonObjectName).englishTerm);
                                break;
                        }
                    }
                }

                return resultVerb;

            default:
                return null;
        }
    }

}
