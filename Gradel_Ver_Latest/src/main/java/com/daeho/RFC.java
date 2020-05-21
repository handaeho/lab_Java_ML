package com.daeho;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;

import java.util.ArrayList;
import java.util.Arrays;

public class RFC {
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
        //inputColsList.remove("target");
        inputColsList.remove("label");
        System.out.println(inputColsList);

        String[] inputCols = inputColsList.parallelStream().toArray(String[]::new);
        System.out.println(inputCols);

        VectorAssembler assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("features");
        Dataset<Row> dataset = assembler.transform(indataset);
        dataset.show();

        Dataset<Row>[] set = dataset.randomSplit(new double[] {0.8, 0.2}, 421);
        Dataset<Row> train_set = set[0];
        Dataset<Row> test_set = set[1];

        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setFeaturesCol("features");

        RandomForestClassificationModel rf_model = rf.fit(train_set);
        Dataset<Row> prediction = rf_model.transform(test_set);

        Dataset<Row> label_prediction = prediction.select("target", "label", "prediction",
                "features", "probability");
        label_prediction.show();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double acc = evaluator.evaluate(prediction);
        System.out.println("Accuracy = " + acc);

        CrossValidator crossval = new CrossValidator().setEstimator(rf).setEvaluator(evaluator);
        ParamMap[] paramgrid = new ParamGridBuilder()
                .addGrid(rf.numTrees(), new int[] {10, 20, 30, 40, 50})
                .addGrid(rf.maxDepth(), new int[] {5, 10, 15})
                .addGrid(rf.maxBins(), new int[] {8, 16, 32, 64})
                .build();

        crossval.setEstimatorParamMaps(paramgrid);
        crossval.setNumFolds(5);

        CrossValidatorModel cv_model = crossval.fit(train_set);
        Model rf_model_best = cv_model.bestModel();

        Dataset<Row> prediction_best = rf_model_best.transform(test_set);
        prediction_best.select("target", "label", "prediction", "features").show();

        double acc_best = evaluator.evaluate(prediction_best);
        System.out.println("Accuracy Best = " + acc_best);
    }
}