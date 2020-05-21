package com.daeho;

import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.apache.spark.ml.regression.DecisionTreeRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;

import java.util.ArrayList;
import java.util.Arrays;

public class DT {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().appName("ML").config("spark.master", "local").getOrCreate();

        StructType schema = new StructType()
                .add("index", "int")
                .add("cycle", "double").add("ECG_a_P", "double")
                .add("ECG_P_b", "double").add("ECG_b_c", "double")
                .add("ECG_c_Q", "double").add("ECG_Q_R", "double")
                .add("ECG_R_S", "double").add("ECG_S_d", "double")
                .add("ECG_d_e", "double").add("ECG_e_T", "double")
                .add("ECG_T_f", "double").add("ECG_S_T", "double")
                .add("ECG_P_peak", "double").add("ECG_Q_peak", "double")
                .add("ECG_R_peak", "double").add("ECG_S_peak", "double")
                .add("ECG_T_peak", "double").add("ECG_RRI", "double")
                .add("TYPE", "double").add("PR_interval", "double")
                .add("PR_segment", "double").add("QRS_complex", "double")
                .add("ST_segment", "double").add("QT_interval", "double")
                .add("target", "double").add("label", "double");

        Dataset<Row> indataset = spark.read().option("header", "true").schema(schema).csv("dataset_index.csv");

        ArrayList<String> inputColsList = new ArrayList<String>(Arrays.asList(indataset.columns()));
        System.out.println(inputColsList);

        inputColsList.remove("index");
        inputColsList.remove("target");
        //inputColsList.remove("label");
        System.out.println(inputColsList);

        String[] inputCols = inputColsList.parallelStream().toArray(String[]::new);

        VectorAssembler assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("features");
        Dataset<Row> dataset = assembler.transform(indataset);
        dataset.show();

        Dataset<Row>[] set = dataset.randomSplit(new double[] {0.8, 0.2}, 421);
        Dataset<Row> train_set = set[0];
        Dataset<Row> test_set = set[1];

        DecisionTreeRegressor dt = new DecisionTreeRegressor()
                .setFeaturesCol("features")
                .setLabelCol("target")
                .setPredictionCol("prediction")
                .setMaxDepth(10)
                .setMaxBins(10);

        DecisionTreeRegressionModel dt_model = dt.fit(train_set);

        Dataset<Row> prediction = dt_model.transform(test_set);

        prediction.select("target","label", "prediction", "features").show();

        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("target")
                .setPredictionCol("prediction")
                .setMetricName("rmse");

        double rmse = evaluator.evaluate(prediction);
        System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);
        System.out.println("Learned regression tree model:\n" + dt_model.toDebugString());
    }
}