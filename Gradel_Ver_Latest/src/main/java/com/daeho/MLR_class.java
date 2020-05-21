package com.daeho;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class MLR_class {
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
        Dataset<Row> result = pred_best.select("target", "label", "prediction");

        return result;
    }

    public static LogisticRegression LR() {
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(100)
                .setRegParam(0.03)
                .setElasticNetParam(0.1)
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setWeightCol("target");

        return lr;
    }

    public static LogisticRegressionModel fit() {
        LogisticRegression lr = LR();

        LogisticRegressionModel lr_model = lr.fit(train_set);

        return lr_model;
    }

    public static Dataset<Row> predict() {
        LogisticRegressionModel lr_model = fit();

        Dataset<Row> predicted = lr_model.transform(test_set);

        return predicted;
    }

    public static Dataset<Row> tuning() {
        LogisticRegression lr = LR();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        CrossValidator crossval = new CrossValidator().setEstimator(lr).setEvaluator(evaluator);
        ParamMap[] paramgrid = new ParamGridBuilder()
                .addGrid(lr.regParam(), new double[] {0.01, 0.05, 0.3})
                .addGrid(lr.maxIter(), new int[] {100, 200, 300})
                .addGrid(lr.elasticNetParam(), new double[] {0.1, 0.5, 0.9})
                .build();

        crossval.setEstimatorParamMaps(paramgrid);
//        crossval.setNumFolds(5);    // K-Fold 검증

        CrossValidatorModel CV_model = crossval.fit(train_set);
        Model lr_model_best = CV_model.bestModel();

        Dataset<Row> predicted_best = lr_model_best.transform(test_set);

        return predicted_best;
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
