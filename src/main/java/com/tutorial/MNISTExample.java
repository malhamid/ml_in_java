package com.tutorial;

import java.io.File;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MNISTExample {

    private static final Logger log = LoggerFactory.getLogger(MNISTExample.class);

    public static void main(String[] args) throws Exception {
        /*
        In this example, we will build a model to classify MNIST image dataset.
        ML Algorithm: CNN
        FrameworkL: Deeplearning4j
         */
        int nChannels = 1; // One input channel to the model.
        int outputNum = 10; // Number of output channels is 10 (digit 0~10)
        int batchSize = 64; // Batch size is 64
        int nEpochs = 10; // Number of training epochs is 10
        int seed = 111; // Random seed identifier


        log.info("Step1: Loading MNIST dataset");
        // Will create two datasets: one for training and another for testing.
        DataSetIterator MNISTTrain = new MnistDataSetIterator(batchSize,true,seed);
        DataSetIterator MNISTTest = new MnistDataSetIterator(batchSize,false,seed);

        log.info("Building the CNN model");
        // Building the CNN model
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed) // random seed
                .l2(0.0005) // regularization
                .weightInit(WeightInit.XAVIER) // initialization of the weight scheme
                .updater(new Adam(1e-3)) // Setting the optimization algorithm
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        //Setting the stride, the kernel size, and the activation function.
                        .nIn(nChannels)
                        .stride(1,1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX) // downsampling the convolution
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        // Setting the stride, kernel size, and the activation function.
                        .stride(1,1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX) // downsampling the convolution
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                // the final output layer is 28x28 with a depth of 1.
                .setInputType(InputType.convolutionalFlat(28,28,1))
                .build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        // initialize the model weights.
        model.init();

        log.info("Step2: start training the model");
        //Setting a listener every 10 iterations and evaluate on test set on every epoch
        model.setListeners(new ScoreIterationListener(10), new EvaluativeListener(MNISTTest, 1, InvocationType.EPOCH_END));
        // Training the model
        model.fit(MNISTTrain, nEpochs);

        String path = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "lenetmnist.zip");

        log.info("Step3: Saving the trained model into the following folder: " + path);
        model.save(new File(path), true);

        log.info("Model training and saving is complete successfully !!");
    }
}
