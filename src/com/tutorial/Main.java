package com.tutorial;

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
    }
}
