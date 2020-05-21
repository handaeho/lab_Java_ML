package com.daeho;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class RFR_class {
    static train_test_split_Regression set = new train_test_split_Regression();

    static Dataset<Row> train_set = set.split()[0];
    static Dataset<Row> test_set = set.split()[1];

    public static void main(String[] args) {
        double rmse = rmse();
        System.out.println("RMSE = " + rmse);
    }

    public static Dataset<Row> execute() {
        Dataset<Row> pred_result = predict();
        pred_result.select("target", "label", "prediction");

        Dataset<Row> pred_best = tuning();
        Dataset<Row> result = pred_best.select("target", "label", "prediction");

        return result;
    }

    public static RandomForestRegressor RF() {
        RandomForestRegressor rf = new RandomForestRegressor()
                .setLabelCol("target")
                .setFeaturesCol("features");

        return rf;
    }

    public static RandomForestRegressionModel fit() {
        RandomForestRegressor rf = RF();

        RandomForestRegressionModel rf_model = rf.fit(train_set);

        return rf_model;
    }

    public static Dataset<Row> predict() {
        RandomForestRegressionModel rf_model = fit();

        Dataset<Row> predicted = rf_model.transform(test_set);

        return predicted;
    }

    public static Dataset<Row> tuning() {
        RandomForestRegressor rf = RF();

        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("target")
                .setPredictionCol("prediction")
                .setMetricName("rmse");

        CrossValidator crossval = new CrossValidator().setEstimator(rf).setEvaluator(evaluator);
        ParamMap[] paramgrid = new ParamGridBuilder()
                .addGrid(rf.numTrees(), new int[] {5, 10, 20})
                .addGrid(rf.maxDepth(), new int[] {5, 10})
                .addGrid(rf.maxBins(), new int[] {4, 8, 16})
                .build();

        crossval.setEstimatorParamMaps(paramgrid);
        crossval.setNumFolds(5);    // K-Fold 검증

        CrossValidatorModel CV_model = crossval.fit(train_set);
        Model rf_model_best = CV_model.bestModel();

        Dataset<Row> predicted_best = rf_model_best.transform(test_set);

        return predicted_best;
    }

    public static double rmse() {
        Dataset<Row> prediction_best = tuning();

        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("target")
                .setPredictionCol("prediction")
                .setMetricName("rmse");

        double rmse = evaluator.evaluate(prediction_best);

        return rmse;
    }
}
