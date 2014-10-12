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

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.mllib.linalg.Vector;

import scala.Serializable;
import scala.Tuple2;

/**
 * This will encapsulate the logic ensembling based on random partition
 */
public class RandomPartitionedEnSembler implements Serializable {

	private static final long serialVersionUID = -3666105986831973419L;

	/**
	 * This class allows the filtering of a selected partition
	 */
	private final class PartionFilterFunction implements
			Function<Tuple2<Integer, LabeledPoint>, Boolean> {

		private static final long serialVersionUID = 2895249051284874295L;
		private int filter;

		public PartionFilterFunction(int filter) {
			this.filter = filter;
		}

		public Boolean call(Tuple2<Integer, LabeledPoint> in) throws Exception {
			return (in._1.intValue() == filter);
		}
	}

	private List<LogisticRegressionModel> models = new ArrayList<LogisticRegressionModel>();
	private int noofModels = 0;
	private double threshold = 0.5;

	public void setNoofModels(int noofModels) {
		this.noofModels = noofModels;
	}

	public void setThreshold(double threshold) {
		this.threshold = threshold;
	}

	/**
	 * expose a simple method that will allow multiple models to be trained based on the partition
	 * @param data
	 */
	public void train(JavaRDD<LabeledPoint> data) {

		// Identify the members randomly of required no. of partitions
		JavaPairRDD<Integer, LabeledPoint> points = data
				.flatMapToPair(new PairFlatMapFunction<LabeledPoint, Integer, LabeledPoint>() {

					private static final long serialVersionUID = 3787644685293675220L;

					public Iterable<Tuple2<Integer, LabeledPoint>> call(
							LabeledPoint labelPoint) throws Exception {
						List<Tuple2<Integer, LabeledPoint>> list = new ArrayList<Tuple2<Integer, LabeledPoint>>();
						list.add(new Tuple2<Integer, LabeledPoint>((int) (Math
								.random() * noofModels), labelPoint));
						return list;
					}
				});

		// Invoke operation so previous instructions for the PairRDD is evaluated
		points.count();
		points.persist(StorageLevel.MEMORY_AND_DISK());

		for (int i = 0; i < noofModels; i++) {
			JavaPairRDD<Integer, LabeledPoint> modeldata = points
					.filter(new PartionFilterFunction(i));
			LogisticRegressionModel model = LogisticRegressionWithSGD.train(
					modeldata.values().rdd(), 100);
			model.setThreshold(threshold);
			models.add(model);
		}
	}

	/**
	 * Each model will independently make a prediction and final outcome is voted
	 * @param Vector
	 * @return double
	 */
	public double voteAndPredit(Vector v) {
		double zero = 0, one = 0;
		// Vote the prediction
		for (int i = 0; i < noofModels; i++) {
			double predict = models.get(i).predict(v);
			if (predict == 0) {
				zero++;
			} else {
				one++;
			}
		}
		// Evaluate the voting
		if (one > zero) {
			return 1.0;
		} else {
			return 0.0;
		}
	}

	/**
	 * Make multiple predictions
	 * @param labelpoints
	 * @return
	 */
	public JavaRDD<Double> voteAndPredit(JavaRDD<LabeledPoint> labelpoints) {
		return labelpoints.map(new Function<LabeledPoint, Double>() {
			private static final long serialVersionUID = -4629831955936964163L;

			public Double call(LabeledPoint labelpoint) throws Exception {
				return voteAndPredit(labelpoint.features());
			}
		});
	}
}
