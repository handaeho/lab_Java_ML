package com.daeho;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import javax.xml.crypto.Data;

public class MLP_class {
    static train_test_split_Multinomial_Classification set = new train_test_split_Multinomial_Classification();

    static Dataset<Row> train_set = set.split()[0];
    static Dataset<Row> test_set = set.split()[1];

    static int[] layers = new int[] {25, 5, 5, 5};

    public static void main(String[] args) {
        double accuracy = accuracy();
        System.out.println("Accuracy = " + accuracy);
    }

    public static Dataset<Row> execute() {
        Dataset<Row> pred_result = predict();
        pred_result.select("target","label", "prediction").show();

        System.out.println("=========================================");

        Dataset<Row> pred_best = tuning();
        Dataset<Row> result = pred_best.select("target","label", "prediction");

        return result;
    }

    public static MultilayerPerceptronClassifier MLP() {
        MultilayerPerceptronClassifier mlp = new MultilayerPerceptronClassifier()
                .setLayers(layers)
                .setBlockSize(128)
                .setSeed(1234L)
                .setMaxIter(100)
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setPredictionCol("prediction");

        return mlp;
    }

    public static MultilayerPerceptronClassificationModel fit() {
        MultilayerPerceptronClassifier mlp = MLP();

        MultilayerPerceptronClassificationModel mlp_model = mlp.fit(train_set);

        return mlp_model;
    }

    public static Dataset<Row> predict() {
        MultilayerPerceptronClassificationModel mlp_model = fit();

        Dataset<Row> predicted = mlp_model.transform(test_set);

        return predicted;
    }

    public static Dataset<Row> tuning() {
        MultilayerPerceptronClassifier mlp = MLP();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setMetricName("accuracy")
                .setLabelCol("label")
                .setPredictionCol("prediction");

        CrossValidator crossval = new CrossValidator().setEstimator(mlp).setEvaluator(evaluator);
        ParamMap[] paramgrid = new ParamGridBuilder()
                .addGrid(mlp.maxIter(), new int[] {100, 300, 500})
                .build();

        crossval.setEstimatorParamMaps(paramgrid);
        crossval.setNumFolds(5);

        CrossValidatorModel CV_model = crossval.fit(test_set);
        Model best_model = CV_model.bestModel();

        Dataset<Row> prediction_best = best_model.transform(test_set);

        return prediction_best;
    }

    public static double accuracy() {
        Dataset<Row> prediction_best = tuning();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setMetricName("accuracy")
                .setLabelCol("label")
                .setPredictionCol("prediction");

        double acc_best = evaluator.evaluate(prediction_best);

        return acc_best;
    }
}
