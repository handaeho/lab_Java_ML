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
/*
>> label Ver.
+------+-----+------------------+--------------------+
|target|label|        prediction|            features|
+------+-----+------------------+--------------------+
|  37.2|  1.0|1.0320512820512822|[0.03,0.0,0.008,0...|
|  10.8|  0.0|               0.0|[0.03,0.021,0.008...|
|  61.0|  3.0|               3.0|[0.03,0.03,0.0,0....|
|  89.1|  4.0|               4.0|[0.03,0.03,0.0,0....|
|   3.3|  0.0|               0.0|[0.03,0.03,0.019,...|
|  31.7|  1.0|               1.0|[0.03,0.03,0.019,...|
|  53.4|  2.0|               2.0|[0.03,0.03,0.02,0...|
|  74.7|  3.0|               3.0|[0.03,0.03,0.02,0...|
|  72.1|  3.0|3.0356536502546687|[0.03,0.03,0.034,...|
|   0.1|  0.0|               0.0|[0.03,0.03,0.0360...|
|  87.7|  4.0|               4.0|[0.03,0.03,0.0360...|
|  97.1|  4.0|               4.0|[0.03,0.03,0.0360...|
|  90.9|  4.0|               4.0|[0.03,0.03,0.0370...|
|  82.1|  4.0|               4.0|[0.03,0.03,0.0370...|
|  57.3|  2.0|               2.2|[0.03,0.03,0.0370...|
|  22.2|  1.0|               1.0|[0.03,0.03,0.0370...|
|  72.5|  3.0|3.1666666666666665|[0.03,0.03,0.038,...|
|   4.6|  0.0|               0.0|[0.03,0.03,0.038,...|
|  77.5|  3.0|3.0356536502546687|[0.03,0.03,0.038,...|
|  99.7|  4.0|               4.0|[0.03,0.03,0.038,...|
+------+-----+------------------+--------------------+
only showing top 20 rows

>> target Ver.
+------+-----+------------------+--------------------+
|target|label|        prediction|            features|
+------+-----+------------------+--------------------+
|  37.2|  1.0|30.406122448979595|[0.03,0.0,0.008,0...|
|  10.8|  0.0|10.485034013605441|[0.03,0.021,0.008...|
|  61.0|  3.0|              69.6|[0.03,0.03,0.0,0....|
|  89.1|  4.0|              87.4|[0.03,0.03,0.0,0....|
|   3.3|  0.0|              5.58|[0.03,0.03,0.019,...|
|  31.7|  1.0|36.400000000000006|[0.03,0.03,0.019,...|
|  53.4|  2.0| 50.91428571428572|[0.03,0.03,0.02,0...|
|  74.7|  3.0| 70.39999999999998|[0.03,0.03,0.02,0...|
|  72.1|  3.0|             74.75|[0.03,0.03,0.034,...|
|   0.1|  0.0| 9.077777777777778|[0.03,0.03,0.0360...|
|  87.7|  4.0|              86.0|[0.03,0.03,0.0360...|
|  97.1|  4.0| 88.74933333333333|[0.03,0.03,0.0360...|
|  90.9|  4.0|              81.2|[0.03,0.03,0.0370...|
|  82.1|  4.0| 84.58333333333333|[0.03,0.03,0.0370...|
|  57.3|  2.0|           48.9448|[0.03,0.03,0.0370...|
|  22.2|  1.0| 31.15666666666667|[0.03,0.03,0.0370...|
|  72.5|  3.0|  69.8320610687023|[0.03,0.03,0.038,...|
|   4.6|  0.0| 9.077777777777778|[0.03,0.03,0.038,...|
|  77.5|  3.0|              64.2|[0.03,0.03,0.038,...|
|  99.7|  4.0| 89.28999999999999|[0.03,0.03,0.038,...|
+------+-----+------------------+--------------------+
only showing top 20 rows

Root Mean Squared Error (RMSE) on test data = 6.15146573040619
*/
