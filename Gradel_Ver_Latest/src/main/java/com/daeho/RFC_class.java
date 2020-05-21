package com.daeho;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class RFC_class {
    static train_test_split_Multinomial_Classification set = new train_test_split_Multinomial_Classification();

    static Dataset<Row> train_set = set.split()[0];
    static Dataset<Row> test_set = set.split()[1];

    public static void main(String[] args) {
        double accuracy = accuracy();
        System.out.println("Accuracy = " + accuracy);
    }

    public static Dataset<Row> execute() {
        Dataset<Row> pred_result = predict();
        pred_result.select("target", "label", "prediction");

        Dataset<Row> pred_best = tuning();
        Dataset<Row> result = pred_best.select("target","label", "prediction");

        return result;
    }

    public static RandomForestClassifier RF() {
        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setFeaturesCol("features");

        return rf;
    }

    public static RandomForestClassificationModel fit() {
        RandomForestClassifier rf = RF();

        RandomForestClassificationModel rf_model = rf.fit(train_set);

        return rf_model;
    }

    public static Dataset<Row> predict() {
        RandomForestClassificationModel rf_model = fit();

        Dataset<Row> predicted = rf_model.transform(test_set);

        return predicted;
    }

    public static Dataset<Row> tuning() {
        RandomForestClassifier rf = RF();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        CrossValidator crossval = new CrossValidator().setEstimator(rf).setEvaluator(evaluator);
        ParamMap[] paramgrid = new ParamGridBuilder()
                .addGrid(rf.numTrees(), new int[] {5, 10, 20})
                .addGrid(rf.maxDepth(), new int[] {5, 10})
                .addGrid(rf.maxBins(), new int[] {4, 8, 16})
                .build();

        crossval.setEstimatorParamMaps(paramgrid);
        crossval.setNumFolds(5);

        CrossValidatorModel cv_model = crossval.fit(train_set);
        Model rf_model_best = cv_model.bestModel();

        Dataset<Row> prediction_best = rf_model_best.transform(test_set);

        return prediction_best;
    }

    public static double accuracy() {
        Dataset<Row> prediction_best = tuning();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double acc_best = evaluator.evaluate(prediction_best);

        return acc_best;
    }
}
