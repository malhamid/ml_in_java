package com.tutorial;



import cc.mallet.types.*;
import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.*;
import cc.mallet.topics.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.net.URL;
import java.util.*;
import java.util.regex.*;
import java.io.*;

public class MalletExample {

    private static final Logger log = LoggerFactory.getLogger(MNISTExample.class);

    public MalletExample() {
    }

    public void buildTopicModel(String inputFile) throws Exception {
         /*
         ap.txt available in the resources folder is the source file we will use to model the document topics.
        Each line represents a document and consists of three attributes separated by comma:
        - Name, label, document content.
         */

        // Creating the pipeline to import the documents from text and map them later to a sequence of features.
        ArrayList<Pipe> pipeList = new ArrayList<Pipe>();

        // Building the pipeline to featurize the input text
        // (1) lowercase words
        // (2) tokenize them
        // (3) remove stopwords
        // (4) map to features
        pipeList.add( new CharSequenceLowercase() );
        pipeList.add( new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")) );
        // Setting the dictionary of the stop words
        URL stopWordsFile = getClass().getClassLoader().getResource("stoplists/en.txt");
        pipeList.add( new TokenSequenceRemoveStopwords(new File(stopWordsFile.toURI()), "UTF-8", false, false, false) );

        pipeList.add( new TokenSequence2FeatureSequence() );
        InstanceList instances = new InstanceList (new SerialPipes(pipeList));

        // Setting the document source file to start the processing
        URL inputFileURL = getClass().getClassLoader().getResource(inputFile);
        Reader fileReader = new InputStreamReader(new FileInputStream(new File(inputFileURL.toURI())), "UTF-8");
        instances.addThruPipe(new CsvIterator (fileReader, Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$"),
                3, 2, 1)); // data, label, name fields

        // Will now create a topic model with the following configuration:
        // No. of topics = 100
        // alpha_t = 1, which control the concentration of topic-word distributions
        // beta_w = 0.01, which represents the pre-word weight over topic-word distributions.
        int numTopics = 100;
        ParallelTopicModel model = new ParallelTopicModel(numTopics, 1.0, 0.01);

        log.info("The number of instances found in the input file is: " + instances.size());
        // adding the instances to the model
        model.addInstances(instances);

        // Here, we can define if multi-thread is used to process the documents providing parallel samplers
        model.setNumThreads(2);

        // defining the number of iterations
        model.setNumIterations(50);
        model.estimate();

        log.info("Displaying the extracted words and topics for the first document");

        // The data alphabet maps word IDs to strings
        Alphabet dataAlphabet = instances.getDataAlphabet();

        // Getting the tokenized features
        FeatureSequence tokens = (FeatureSequence) model.getData().get(0).instance.getData();
        // Getting the extracted topics
        LabelSequence topics = model.getData().get(0).topicSequence;

        Formatter out = new Formatter(new StringBuilder(), Locale.US);
        for (int position = 0; position < tokens.getLength(); position++) {
            out.format("%s-%d ", dataAlphabet.lookupObject(tokens.getIndexAtPosition(position)), topics.getIndexAtPosition(position));
        }
        log.info("--- Displaying the model output for the first document ---");
        log.info(out.toString());

        // Predict the topic for the first document:
        double[] topicDistribution = model.getTopicProbabilities(0);

        // Get an array of sorted sets of word ID/count pairs
        ArrayList<TreeSet<IDSorter>> topicSortedWords = model.getSortedWords();

        // Show top 5 words in topics with proportions for the first document
        for (int topic = 0; topic < numTopics; topic++) {
            Iterator<IDSorter> iterator = topicSortedWords.get(topic).iterator();

            out = new Formatter(new StringBuilder(), Locale.US);
            out.format("%d\t%.3f\t", topic, topicDistribution[topic]);
            int rank = 0;
            while (iterator.hasNext() && rank < 5) {
                IDSorter idCountPair = iterator.next();
                out.format("%s (%.0f) ", dataAlphabet.lookupObject(idCountPair.getID()), idCountPair.getWeight());
                rank++;
            }
            log.info("Displaying the list of sorted topics against the first document: ");
            log.info(out.toString());
        }
    }

    public static void main(String[] args) throws Exception {
        MalletExample exampleRunner = new MalletExample();
        try {
            exampleRunner.buildTopicModel("ap.txt");
        }catch(Exception e) {
            System.out.println(e.getMessage());
        }
    }
}
