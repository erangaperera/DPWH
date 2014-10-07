package org.wso2.carbon.lrtest;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.storage.StorageLevel;

import scala.Serializable;
import scala.Tuple2;

public class HistogramEnsembler implements Serializable{
	
	private static final long serialVersionUID = 5904878364429620103L;
	
	private final class HistogramFilter implements 
			Function<Tuple2<Integer, LabeledPoint>, Boolean> {
		
		private static final long serialVersionUID = 3774797728292061024L;
		private int binGroupFilter;
		
		public HistogramFilter(int binGroup) {
			this.binGroupFilter = binGroup;
		}

		public Boolean call(Tuple2<Integer, LabeledPoint> in) throws Exception {
			int binGroup = binGroups.get(in._1);
			return (binGroup == binGroupFilter);
		}
	}
	
	private Map<Integer, LogisticRegressionModel> models = new HashMap<Integer, LogisticRegressionModel>();
	private int noofBins = 0;
	private IHistogramHelper histogramHelper;
	private static HistogramTree histogramTree;
	private double threshold = 0.5;
	private Map<Integer, Integer> binGroups = new HashMap<Integer, Integer>();

	public void setThreshold(double threshold) {
		this.threshold = threshold;
	}

	public HistogramEnsembler(IHistogramHelper histogramHelper, int noofBins) {
		this.noofBins = noofBins;
		this.histogramHelper = histogramHelper;
		histogramTree = new HistogramTree(histogramHelper);
	}
	
	public void train(JavaRDD<LabeledPoint> data){
		
		JavaPairRDD<Integer, LabeledPoint> binMappedData = data.mapToPair(new PairFunction<LabeledPoint, Integer, LabeledPoint>() {

			private static final long serialVersionUID = 5789091614872051176L;

			public Tuple2<Integer, LabeledPoint> call(LabeledPoint labelPoint)
					throws Exception {
				int binNo = histogramHelper.getBinNo(labelPoint.features());
				histogramTree.AddNode(binNo, 1);
				return new Tuple2<Integer, LabeledPoint>(binNo, labelPoint);
			}
		});
		
		// Invoke operation so PairRDD is evaluated
		binMappedData.count();
		binMappedData.persist(StorageLevel.MEMORY_AND_DISK());

		Map<Integer, Integer[]> binGroupArray = histogramTree.groupBins(noofBins);
		
		// Sometime the actual no of bin will be varied from the desired
		noofBins = binGroupArray.size();
		for (int i = 0; i < noofBins; i++) {
			Integer[] temp = binGroupArray.get(i);
			for (int j = 0; j < temp.length; j++) {
				binGroups.put(temp[j], i);
			}
		}
		
		for (int i = 0; i < noofBins; i++) {
			JavaPairRDD<Integer, LabeledPoint> modeldata = binMappedData.filter(new HistogramFilter(i));
			if (modeldata.count() > 0){
				LogisticRegressionModel model = LogisticRegressionWithSGD.train(modeldata.values().rdd(), 100); 
				model.setThreshold(threshold);
				models.put(i, model);
			}
		}
	}

	public double predict(Vector v){
		
		int binNo = histogramHelper.getBinNo(v);
		int binGroup = binGroups.get(binNo);
//		if (binGroups.containsKey(binNo))
//			binGroup = binGroups.get(binNo);
//		else
//			return 0.0;
		return models.get(binGroup).predict(v);
	}
	
	public JavaRDD<Double> predit(JavaRDD<LabeledPoint> labelpoints){
		
		return labelpoints.map(new Function<LabeledPoint, Double>() {
			private static final long serialVersionUID = 3575346820732202547L;
			public Double call(LabeledPoint labelpoint) throws Exception {
				return predict(labelpoint.features());
			}
		});
	}
}
