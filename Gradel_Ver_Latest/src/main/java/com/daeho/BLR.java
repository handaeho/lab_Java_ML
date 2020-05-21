package com.daeho;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
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

public class BLR {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().appName("ML").config("spark.master", "local").getOrCreate();

        StructType schema = new StructType()
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
                .add("OX", "double");

        Dataset<Row> indataset = spark.read().option("header", "true").schema(schema).csv("dataset_OX.csv");

        ArrayList<String> inputColsList = new ArrayList<String>(Arrays.asList(indataset.columns()));
        System.out.println(inputColsList);

        //inputColsList.remove("target");
        inputColsList.remove("OX");
        System.out.println(inputColsList);

        String[] inputCols = inputColsList.parallelStream().toArray(String[]::new);

        VectorAssembler assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("features");
        Dataset<Row> dataset = assembler.transform(indataset);
        dataset.show();

        Dataset<Row>[] set = dataset.randomSplit(new double[] {0.8, 0.2}, 421);
        Dataset<Row> train_set = set[0];
        Dataset<Row> test_set = set[1];

        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(100)
                .setRegParam(0.03)
                .setLabelCol("OX")
                .setFeaturesCol("features");

        // Fit the model
        LogisticRegressionModel lrModel = lr.fit(train_set);

        Dataset<Row> predicted = lrModel.transform(test_set);
        predicted.select("OX", "prediction", "features").show();

        // Print the coefficients and intercept for multinomial logistic regression
        System.out.println("Coefficients: \n"
                + lrModel.coefficientMatrix() + " \nIntercept: " + lrModel.interceptVector());
        LogisticRegressionTrainingSummary trainingSummary = lrModel.summary();

        double accuracy = trainingSummary.accuracy();
        System.out.println(accuracy);

        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator()
                .setLabelCol("OX")
                .setRawPredictionCol("rawPrediction")
                .setMetricName("areaUnderROC");

        double areaUnderROC = evaluator.evaluate(predicted);
        System.out.println("areaUnderROC on test data = " + areaUnderROC);

        CrossValidator crossval = new CrossValidator().setEstimator(lr).setEvaluator(evaluator);
        ParamMap[] paramgrid = new ParamGridBuilder()
                .addGrid(lr.regParam(), new double[] {0.01, 0.03, 0.05, 0.3, 0.5})
                .addGrid(lr.maxIter(), new int[] {100, 200, 300, 500})
                .build();

        crossval.setEstimatorParamMaps(paramgrid);
        crossval.setNumFolds(5);    // K-Fold 검증

        CrossValidatorModel CV_model = crossval.fit(train_set);
        Model lr_model_best = CV_model.bestModel();

        Dataset<Row> predicted_best = lr_model_best.transform(test_set);
        predicted_best.select("OX", "prediction", "features").show();

        double areaUnderROC_best = evaluator.evaluate(predicted_best);
        System.out.println(areaUnderROC_best);
    }
}

