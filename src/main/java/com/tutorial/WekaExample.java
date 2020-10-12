package com.tutorial;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaExample {

    private static final Logger log = LoggerFactory.getLogger(MNISTExample.class);

    public static void main(String[] args) throws Exception {
        // Specifying the datasource
        DataSource dataSource = new DataSource("data.arff");
        // Loading the dataset
        Instances dataInstances = dataSource.getDataSet();
        // Displaying the number of instances
        log.info("The number of loaded instances is: " + dataInstances.numInstances());

        log.info("data:" + dataInstances.toString());

        // Identifying the label index
        dataInstances.setClassIndex(dataInstances.numAttributes() - 1);

        log.info("The number of attributes in the dataset: " + dataInstances.numAttributes());
        log.info("The number of classes: " + dataInstances.numClasses());

        // printing the column name
        for(int i=0; i < dataInstances.numAttributes(); i++){
            log.info("Column " + i + "->" + dataInstances.attribute(i));
        }

        // Creating a decision free classifier
        J48 treeClassifier = new J48();

        treeClassifier.setOptions(new String[] { "-U" });

        treeClassifier.buildClassifier(dataInstances);

        System.out.println(treeClassifier);

    }
}
