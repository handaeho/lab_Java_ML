package com.daeho;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.classification.LinearSVCModel;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class SVM_class {
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

    public static LinearSVC LSVC() {
        LinearSVC lsvc = new LinearSVC()
                .setMaxIter(100)
                .setRegParam(0.03)
                .setLabelCol("OX")
                .setFeaturesCol("features");

        return lsvc;
    }

    public static LinearSVCModel fit() {
        LinearSVC lsvc = LSVC();

        // Fit the model
        LinearSVCModel lsvc_model = lsvc.fit(train_set);

        return lsvc_model;
    }

    public static Dataset<Row> predict() {
        LinearSVCModel lsvc_Model = fit();

        Dataset<Row> predicted = lsvc_Model.transform(test_set);

        return predicted;
    }

    public static Dataset<Row> tuning() {
        LinearSVC lsvc = LSVC();

        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator()
                .setLabelCol("OX")
                .setRawPredictionCol("rawPrediction")
                .setMetricName("areaUnderROC");

        CrossValidator crossval = new CrossValidator().setEstimator(lsvc).setEvaluator(evaluator);
        ParamMap[] paramgrid = new ParamGridBuilder()
                .addGrid(lsvc.maxIter(), new int[] {100, 200, 300})
                .addGrid(lsvc.regParam(), new double[] {0.01, 0.05, 0.3})
                .build();

        crossval.setEstimatorParamMaps(paramgrid);
        crossval.setNumFolds(5);

        CrossValidatorModel cv_model = crossval.fit(train_set);
        Model lsvm_model_best = cv_model.bestModel();

        Dataset<Row> prediction_best = lsvm_model_best.transform(test_set);

        return prediction_best;
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
