import org.apache.spark.ml.Model;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;

import java.util.ArrayList;
import java.util.Arrays;

public class RFR {
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
                .add("target", "double").add("label", "double");

        Dataset<Row> indataset = spark.read().option("header", "true").schema(schema).csv("dataset.csv");

        ArrayList<String> inputColsList = new ArrayList<String>(Arrays.asList(indataset.columns()));
        System.out.println(inputColsList);

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

        RandomForestRegressor rf = new RandomForestRegressor()
                .setLabelCol("target")
                .setFeaturesCol("features");

        RandomForestRegressionModel rf_model = rf.fit(train_set);
        Dataset<Row> prediction = rf_model.transform(test_set);

        Dataset<Row> label_prediction = prediction.select("target", "label", "prediction", "features");
        label_prediction.show();

        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("target")
                .setPredictionCol("prediction")
                .setMetricName("rmse");

        double rmse = evaluator.evaluate(prediction);
        System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);

        CrossValidator crossval = new CrossValidator().setEstimator(rf).setEvaluator(evaluator);
        ParamMap[] paramgrid = new ParamGridBuilder()
                .addGrid(rf.numTrees(), new int[] {10, 20, 30, 50})
                .addGrid(rf.maxDepth(), new int[] {5, 10, 15})
                .addGrid(rf.maxBins(), new int[] {8, 16, 32, 64})
                .build();

        crossval.setEstimatorParamMaps(paramgrid);
        crossval.setNumFolds(2);    // K-Fold 검증

        CrossValidatorModel CV_model = crossval.fit(train_set);
        Model rf_model_best = CV_model.bestModel();

        Dataset<Row> predicted_best = rf_model_best.transform(test_set);
        predicted_best.select("target", "label", "prediction", "features").show();

        double acc_best = evaluator.evaluate(predicted_best);
        System.out.println(acc_best);

    }
}
/*
+------+-----+------------------+--------------------+
|target|label|        prediction|            features|
+------+-----+------------------+--------------------+
|  37.2|  1.0| 36.24206572122433|[0.03,0.0,0.008,0...|
|  10.8|  0.0| 10.36031797452787|[0.03,0.021,0.008...|
|  61.0|  3.0| 62.72338095238095|[0.03,0.03,0.0,0....|
|  89.1|  4.0| 89.47319764325312|[0.03,0.03,0.0,0....|
|   3.3|  0.0|7.0974614793671345|[0.03,0.03,0.019,...|
|  31.7|  1.0| 33.99284703205451|[0.03,0.03,0.019,...|
|  53.4|  2.0|54.560586343008545|[0.03,0.03,0.02,0...|
|  74.7|  3.0| 71.89652752531816|[0.03,0.03,0.02,0...|
|  72.1|  3.0| 71.90582677046731|[0.03,0.03,0.034,...|
|   0.1|  0.0| 3.420333077048673|[0.03,0.03,0.0360...|
|  87.7|  4.0| 87.98621216216239|[0.03,0.03,0.0360...|
|  97.1|  4.0| 92.51948702811387|[0.03,0.03,0.0360...|
|  90.9|  4.0| 91.87733554283484|[0.03,0.03,0.0370...|
|  82.1|  4.0| 82.59706052819014|[0.03,0.03,0.0370...|
|  57.3|  2.0|57.201289211683076|[0.03,0.03,0.0370...|
|  22.2|  1.0|21.049933128455802|[0.03,0.03,0.0370...|
|  72.5|  3.0|  71.2606119464613|[0.03,0.03,0.038,...|
|   4.6|  0.0|5.1868728493565035|[0.03,0.03,0.038,...|
|  77.5|  3.0| 78.80940443585443|[0.03,0.03,0.038,...|
|  99.7|  4.0| 95.77761991826104|[0.03,0.03,0.038,...|
+------+-----+------------------+--------------------+
only showing top 20 rows

1.4147762467398723
*/