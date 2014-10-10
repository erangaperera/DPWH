/*
 * Copyright (c) 2005-2014, WSO2 Inc. (http://www.wso2.org) All Rights Reserved.
 *
 * WSO2 Inc. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.wso2.carbon.lrtest;

import java.util.HashMap;
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

/**
 *	This will encapsulate the logic for Histogram aided ensembling
 */
public class HistogramEnsembler implements Serializable {

	private static final long serialVersionUID = 5904878364429620103L;
	private Map<Integer, LogisticRegressionModel> models = new HashMap<Integer, LogisticRegressionModel>();
	private int noofBins = 0;
	private IHistogramHelper histogramHelper;
	private static HistogramTree histogramTree;
	private double threshold = 0.5;
	private int[] binGroups;	

	
	/**
	 *	Provides filtering for a JavaPairRDD who belongs to the specified group
	 */
	private final class HistogramFilter implements
			Function<Tuple2<Integer, LabeledPoint>, Boolean> {

		private static final long serialVersionUID = 3774797728292061024L;
		private int binGroupFilter;

		public HistogramFilter(int binGroup) {
			this.binGroupFilter = binGroup;
		}

		public Boolean call(Tuple2<Integer, LabeledPoint> in) throws Exception {
			int binGroup = binGroups[in._1];
			return (binGroup == binGroupFilter);
		}
	}

	public void setThreshold(double threshold) {
		this.threshold = threshold;
	}

	/**
	 * A histogram helper implementation should be passed with the constructor.
	 * 
	 * @param histogramHelper
	 * @param noofBins
	 */
	public HistogramEnsembler(IHistogramHelper histogramHelper, int noofBins) {
		this.noofBins = noofBins;
		this.histogramHelper = histogramHelper;
		histogramTree = new HistogramTree(histogramHelper);
	}

	public void train(JavaRDD<LabeledPoint> data) {

		// Create all possible bins on the tree to avoid
		// Inconsistencies within groups identified in training 
		// dataset and test dataset
		int noofDimensions = histogramHelper.getNoofDimensions();
		int reqNoofBins = 1;
		for (int i = 0; i < noofDimensions; i++) {
			reqNoofBins = reqNoofBins * histogramHelper.getSplits(i);
		}
		for (int i = 0; i < reqNoofBins; i++) {
			histogramTree.AddNode(i, 0);
		}

		// Construct the JavaPairRDD and Histogram simultaneously
		JavaPairRDD<Integer, LabeledPoint> binMappedData = data
				.mapToPair(new PairFunction<LabeledPoint, Integer, LabeledPoint>() {

					private static final long serialVersionUID = 5789091614872051176L;

					public Tuple2<Integer, LabeledPoint> call(
							LabeledPoint labelPoint) throws Exception {
						int binNo = histogramHelper.getBinNo(labelPoint
								.features());
						histogramTree.AddNode(binNo, 1);
						return new Tuple2<Integer, LabeledPoint>(binNo,
								labelPoint);
					}
				});

		// Invoke operation so PairRDD is evaluated
		binMappedData.count();
		binMappedData.persist(StorageLevel.MEMORY_AND_DISK());

		// Group the bin into the number specified
		Map<Integer, Integer[]> binGroupArray = histogramTree
				.groupBins(noofBins);

		// Sometime the actual no of bin will be varied from the desired, 
		// therefore read back the actual no. of groups
		noofBins = binGroupArray.size();
		binGroups = new int[reqNoofBins];
		for (int i = 0; i < noofBins; i++) {
			Integer[] temp = binGroupArray.get(i);
			for (int j = 0; j < temp.length; j++) {
				binGroups[temp[j]] = i;
			}
		}

		// For each group train a specialized model
		for (int i = 0; i < noofBins; i++) {
			JavaPairRDD<Integer, LabeledPoint> modeldata = binMappedData
					.filter(new HistogramFilter(i));
			if (modeldata.count() > 0) {
				LogisticRegressionModel model = LogisticRegressionWithSGD
						.train(modeldata.values().rdd(), 100);
				model.setThreshold(threshold);
				models.put(i, model);
			}
		}
	}

	/**
	 * Predict a single value
	 * 
	 * @param v
	 * @return double
	 */
	public double predict(Vector v) {

		// Identify and pick the correct model
		int binNo = histogramHelper.getBinNo(v);
		int binGroup = binGroups[binNo];
		return models.get(binGroup).predict(v);
		
	}

	/**
	 * Predict multiple values
	 * 
	 * @param labelpoints
	 * @return
	 */
	public JavaRDD<Double> predit(JavaRDD<LabeledPoint> labelpoints) {

		return labelpoints.map(new Function<LabeledPoint, Double>() {
			private static final long serialVersionUID = 3575346820732202547L;

			public Double call(LabeledPoint labelpoint) throws Exception {
				return predict(labelpoint.features());
			}
		});
	}
}
