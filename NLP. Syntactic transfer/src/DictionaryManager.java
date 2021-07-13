import org.json.JSONObject;
import java.io.*;
import java.util.HashMap;

//This class manages the word translation from Italian to English
public class DictionaryManager {

    HashMap<String, DictionaryEntry> DictionaryEntries;

    public DictionaryManager(String pathDictionary) throws IOException {

        //Initializing dictionary entries reading from a file
        DictionaryEntries = new HashMap<>();

        BufferedReader dictionaryFile = new BufferedReader(new FileReader(pathDictionary));
        String line;
        while ((line = dictionaryFile.readLine()) != null){

            //getting values from line
            String[] values = line.split("=");
            String[] dictionaryEntryValues = values[1].split(";");

            //setting entry values
            DictionaryEntry entry = new DictionaryEntry();
            entry.englishTerm = dictionaryEntryValues[0];

            //set plural feature
            if (dictionaryEntryValues.length > 1)
                entry.isPlural = dictionaryEntryValues[1].equals("plural");

            DictionaryEntries.put(values[0], entry);
        }

        dictionaryFile.close();
    }

    //This method translated an italian term to english
    public DictionaryEntry translateTerm(JSONObject object, String name){

        //get italian term from json object
        String italianTerm = object.getString(name);

        DictionaryEntry result = DictionaryEntries.get(italianTerm.toLowerCase());

        //If we don't get the element, we try without considering the lower case
        if(result == null)
            return DictionaryEntries.get(italianTerm);
        else
            return result;
    }

    //This method indicates whether the agent of the verb is a person
    public boolean flagAgentVerb(JSONObject object, String name){

        String italianTerm = object.getString(name);

        //The agent of the verb is a person if the verb match a specific word
        //We did so because of the simplicity of the case
        return italianTerm.equals("fatto");
    }

    //English dictionary entry
    public static class DictionaryEntry
    {
        public String englishTerm;
        public boolean isPlural;
    }
}
