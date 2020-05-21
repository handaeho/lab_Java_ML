package com.daeho;

import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.apache.spark.ml.regression.DecisionTreeRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class DT_class {
    static train_test_split_Multinomial_Classification set = new train_test_split_Multinomial_Classification();

    static Dataset<Row> train_set = set.split()[0];
    static Dataset<Row> test_set = set.split()[1];

    public static void main(String[] args) {
        double RMSE = evaluate();
        System.out.println("RSME = " + RMSE);
    }

    public static Dataset<Row> execute() {
        Dataset<Row> pred_result = predict();
        Dataset<Row> result = pred_result.select("target","label", "prediction");

        return result;
    }

    public static DecisionTreeRegressor DT() {
        DecisionTreeRegressor dt = new DecisionTreeRegressor()
                .setFeaturesCol("features")
                .setLabelCol("target")
                .setPredictionCol("prediction")
                .setMaxDepth(10)
                .setMaxBins(10);

        return dt;
    }

    public static DecisionTreeRegressionModel fit() {
        DecisionTreeRegressor dt = DT();

        // Fit the model
        DecisionTreeRegressionModel dt_model = dt.fit(train_set);

        return dt_model;
    }

    public static Dataset<Row> predict() {
        DecisionTreeRegressionModel dt_model = fit();

        Dataset<Row> predicted = dt_model.transform(test_set);

        return predicted;
    }

    public static double evaluate() {
        Dataset<Row> predictied = predict();

        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("target")
                .setPredictionCol("prediction")
                .setMetricName("rmse");

        double rmse = evaluator.evaluate(predictied);

        return rmse;
    }
}
