package com.daeho;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class BLR_class {
    static train_test_split_Binary_Classification set = new train_test_split_Binary_Classification();

    static Dataset<Row> train_set = set.split()[0];
    static Dataset<Row> test_set = set.split()[1];

    public static void main(String[] args) {
        double UnderROC_rate = under_ROC_rate();
        System.out.println("UnderROC_rate = " + UnderROC_rate);
    }

    public static Dataset<Row> execute() {
        Dataset<Row> pred_result = predict();
        pred_result.select("OX", "prediction");

        Dataset<Row> pred_best = tuning();
        Dataset<Row> result = pred_best.select("OX", "prediction");

        return result;
    }

    public static LogisticRegression LR() {
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(100)
                .setRegParam(0.03)
                .setLabelCol("OX")
                .setFeaturesCol("features");

        return lr;
    }

    public static LogisticRegressionModel fit() {
        LogisticRegression lr = LR();

        // Fit the model
        LogisticRegressionModel lr_Model = lr.fit(train_set);

        return lr_Model;
    }

    public static Dataset<Row> predict() {
        LogisticRegressionModel lr_Model = fit();

        Dataset<Row> predicted = lr_Model.transform(test_set);

        return predicted;
    }

    public static Dataset<Row> tuning() {
        LogisticRegression lr = LR();

        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator()
                .setLabelCol("OX")
                .setRawPredictionCol("rawPrediction")
                .setMetricName("areaUnderROC");

        CrossValidator crossval = new CrossValidator().setEstimator(lr).setEvaluator(evaluator);
        ParamMap[] paramgrid = new ParamGridBuilder()
                .addGrid(lr.regParam(), new double[] {0.01, 0.05, 0.3})
                .addGrid(lr.maxIter(), new int[] {100, 200, 300})
                .build();

        crossval.setEstimatorParamMaps(paramgrid);
        crossval.setNumFolds(5);    // K-Fold 검증

        CrossValidatorModel CV_model = crossval.fit(train_set);
        Model lr_model_best = CV_model.bestModel();

        Dataset<Row> predicted_best = lr_model_best.transform(test_set);

        return predicted_best;
    }

    public static double under_ROC_rate() {
        Dataset<Row> predicted_best = tuning();

        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator()
                .setLabelCol("OX")
                .setRawPredictionCol("rawPrediction")
                .setMetricName("areaUnderROC");

        double areaUnderROC_best = evaluator.evaluate(predicted_best);

        return areaUnderROC_best;
    }
}
