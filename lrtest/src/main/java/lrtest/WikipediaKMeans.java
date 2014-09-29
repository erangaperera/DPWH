package lrtest;

import org.apache.log4j.Logger;
import org.apache.log4j.Level;

import java.util.*;

import com.google.common.collect.Lists;

import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.util.Vector;
 
 @SuppressWarnings("deprecation")
public class WikipediaKMeans {
   static int closestPoint(Vector p, List<Vector> centers) {
     int bestIndex = 0;
     double closest = Double.POSITIVE_INFINITY;
     for (int i = 0; i < centers.size(); i++) {
       double tempDist = p.squaredDist(centers.get(i));
       if (tempDist < closest) {
         closest = tempDist;
         bestIndex = i;
       }
     }
     return bestIndex;
   }
   static org.apache.spark.util.Vector average(Iterable<Vector> ps) {
     int numVectors = 0; // = ps.size();
     Vector out = null; // = ps.iterator().next();
     for (Vector vector : ps) {
    	 if (numVectors == 0){
    		 out = vector;
    	 }
    	 numVectors += 1;
    	 out.addInPlace(vector);
	}
    return out.divide(numVectors);
   }

@SuppressWarnings("serial")
public static void main(String[] args) throws Exception {
     Logger.getLogger("spark").setLevel(Level.WARN);
     //String sparkHome = "/root/spark";
     //String jarFile = "target/scala-2.9.3/wikipedia-kmeans_2.9.3-0.0.jar";
     //String master = JavaHelpers.getSparkUrl();
     //String masterHostname = JavaHelpers.getMasterHostname();
     JavaSparkContext sc = new JavaSparkContext("local", /* master,*/ "WikipediaKMeans"); /*,
       sparkHome, jarFile); */
     int K = 2;
     double convergeDist = .000001;
     JavaPairRDD<String, Vector> data = sc.textFile("/Users/erangap/Documents/ML_Project/Test2.txt").mapToPair(
       new PairFunction<String, String, Vector>() {
         public Tuple2<String, Vector> call(String in) throws Exception {
           String[] parts = in.split("#");
           return new Tuple2<String, Vector>(
            parts[0], JavaHelpers.parseVector(parts[1]));
         }
       }).cache();
     long count = data.count();
     System.out.println("Number of records " + count);
     List<Tuple2<String, Vector>> centroidTuples = data.takeSample(false, K, 42);
     final List<Vector> centroids = Lists.newArrayList();
     for (Tuple2<String, Vector> t: centroidTuples) {
       centroids.add(t._2());
     }
     System.out.println("Done selecting initial centroids");
     double tempDist;
     do {
       JavaPairRDD<Integer, Vector> closest = data.mapToPair(
         new PairFunction<Tuple2<String, Vector>, Integer, Vector>() {
           public Tuple2<Integer, Vector> call(Tuple2<String, Vector> in) throws Exception {
             return new Tuple2<Integer, Vector>(closestPoint(in._2(), centroids), in._2());
           }
         }
       );
       JavaPairRDD<Integer, Iterable<Vector>> pointsGroup = closest.groupByKey();
       
       Map<Integer, Vector> newCentroids = pointsGroup.mapValues(new Function<Iterable<Vector>, Vector>() {
	   		public Vector call(Iterable<Vector> ps) throws Exception {
	   			// TODO Auto-generated method stub
	   			return average(ps);
	   		}
   		}).collectAsMap();
       tempDist = 0.0;
       for (int i = 0; i < K; i++) {
         tempDist += centroids.get(i).squaredDist(newCentroids.get(i));
       }
       for (Map.Entry<Integer, Vector> t: newCentroids.entrySet()) {
         centroids.set(t.getKey(), t.getValue());
       }
       System.out.println("Finished iteration (delta = " + tempDist + ")");
     } while (tempDist > convergeDist);
     System.out.println("Cluster with some articles:");
     int numArticles = 3;
     for (int i = 0; i < centroids.size(); i++) {
       final int index = i;
       List<Tuple2<String, Vector>> samples =
       data.filter(new Function<Tuple2<String, Vector>, Boolean>() {
         public Boolean call(Tuple2<String, Vector> in) throws Exception {
         return closestPoint(in._2(), centroids) == index;
       }}).take(numArticles);
       for(Tuple2<String, Vector> sample: samples) {
        System.out.println(sample._1());
       }
       System.out.println();
     }
     sc.stop();
     System.exit(0);
   }
 }