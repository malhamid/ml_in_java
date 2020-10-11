package com.tutorial;

import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {

    public static void main(String[] args) throws Exception {
        // Specifying the datasource
        DataSource dataSource = new DataSource("data.arff");
        // Loading the dataset
        Instances dataInstances = dataSource.getDataSet();
        // Displaying the number of instances
        System.out.println("The number of loaded instances is: " + dataInstances.numInstances());

        System.out.println("data:" + dataInstances.toString());

        // Identifying the label index
        dataInstances.setClassIndex(dataInstances.numAttributes() - 1);

        System.out.println("The number of attributes in the dataset: " + dataInstances.numAttributes());
        System.out.println("The number of classes: " + dataInstances.numClasses());

        // printing the column name
        for(int i=0; i < dataInstances.numAttributes(); i++){
            System.out.println("Column " + i + "->" + dataInstances.attribute(i));
        }

        // Creating a decision free classifier
        J48 treeClassifier = new J48();

        treeClassifier.setOptions(new String[] { "-U" });

        treeClassifier.buildClassifier(dataInstances);

        System.out.println(treeClassifier);

    }
}
