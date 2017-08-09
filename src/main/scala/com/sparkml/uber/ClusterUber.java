package com.lynda.course.sparkbde;

// $example on$
import java.io.IOException;
import java.io.Serializable;
import java.sql.Timestamp;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
// $example off$
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.TimestampType;

/**
 * An example demonstrating k-means clustering. Run with
 * 
 * <pre>
 * bin/run-example ml.JavaKMeansExample
 * </pre>
 */
public class ClusterUber {
	public static class Uber implements Serializable {
		private Timestamp date;
		private double latitude;
		private double longitude;
		private String base;

		public Timestamp getDate() {
			return date;
		}

		public void setDate(Timestamp date) {
			this.date = date;
		}

		public double getLatitude() {
			return latitude;
		}

		public void setLatitude(double latitude) {
			this.latitude = latitude;
		}

		public double getLongitude() {
			return longitude;
		}

		public void setLongitude(double longitude) {
			this.longitude = longitude;
		}

		public String getBase() {
			return base;
		}

		public void setBase(String base) {
			this.base = base;
		}

	}

	public static void main(String[] args) throws IOException {
		SparkConf conf = new SparkConf().setMaster("local").setAppName(
				"Work Count App");

		// Create a Java version of the Spark Context from the configuration
		JavaSparkContext sc = new JavaSparkContext(conf);

		// Create a SparkSession.
		SparkSession spark = SparkSession.builder()
				.appName("JavaKMeansExample").getOrCreate();

		// Create an RDD of Person objects from a text file
		JavaRDD<Uber> uberRDD = spark.read().textFile("data/uber.csv")
				.javaRDD().map(line -> {
					String[] parts = line.split(",");
					Uber uber = new Uber();
					uber.setDate(Timestamp.valueOf(parts[0]));
					uber.setLatitude(Double.parseDouble(parts[1]));
					uber.setLongitude(Double.parseDouble(parts[2]));
					uber.setBase(parts[3]);
					return uber;
				});

		// Apply a schema to an RDD of JavaBeans to get a DataFrame
		Dataset<Row> uberDF = spark.createDataFrame(uberRDD, Uber.class);

		// uberDF.cache();
		// uberDF.show();
		 //uberDF.printSchema();

		String[] featureCols = { "latitude", "longitude" };

		VectorAssembler assembler = new VectorAssembler().setInputCols(
				featureCols).setOutputCol("features");
		Dataset<Row> uberDFOther = assembler.transform(uberDF);
		//uberDFOther.show();

		Dataset<Row>[] trainingAndTestData = uberDFOther.randomSplit(
				new double[] { 0.7, 03 }, 5043);
		KMeans kmeans = new KMeans().setK(10).setFeaturesCol("features").setPredictionCol("prediction")
				.setMaxIter(5);

		KMeansModel model = kmeans.fit(trainingAndTestData[0]);
		Vector[] centers = model.clusterCenters();
//		System.out.println("Cluster Centers: ");
//		for (Vector center : centers) {
//			System.out.println(center);
//		}
	    Dataset<Row> categories = model.transform(trainingAndTestData[1]);
	  
	    //categories.show();
	    
	    categories.registerTempTable("uber");
	    categories.createOrReplaceTempView("uber");
	    //"Which hours of the day and which cluster had the highest number of pickups?"
//	  Dataset<Row> sqlDF = spark.sql("SELECT HOUR(uber.date) as hour, prediction, count(uber.prediction) as count FROM uber GROUP BY HOUR(uber.date), prediction ORDER BY count(uber.prediction) DESC");
//	    sqlDF.show();
	    
	    //How many pickups occurred in each cluster?
//	    Dataset<Row> pickUpDF = spark.sql("SELECT  prediction, count(prediction) as count FROM uber GROUP BY  prediction ");
//	    pickUpDF.show();
	    


	    // to save the categories dataframe as json data
	      categories.select("date", "base", "prediction").write().format("json").save("uberclusterstest");
	    //  to save the model 
	      model.write().overwrite().save("data/savemodel");
	    //  to re-load the model
	    KMeansModel sameModel = KMeansModel.load("data/savemodel");
	    
	    
	    
	    


		spark.stop();
	}
}