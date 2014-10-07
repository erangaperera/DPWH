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

import java.io.Serializable;
import java.util.ArrayList;


//import org.apache.spark.util.Vector;
import org.apache.spark.mllib.linalg.Vector;

/**
 * The Implementation of the HistogramHelper
 * @author erangap@wso2.com
 */
public class HistogramHelper implements Serializable, IHistogramHelper {
	
	private static final long serialVersionUID = 3116348398807816316L;
	
	private final ArrayList<Integer> dimensions = new ArrayList<Integer>();
	private final ArrayList<Double> minValues = new ArrayList<Double>();
	private final ArrayList<Double> maxValues = new ArrayList<Double>();
	private final ArrayList<Integer> splitSizes = new ArrayList<Integer>();
	private final ArrayList<Integer> multiplier = new ArrayList<Integer>();
	
	/**
	 * Add one dimension at a time
	 * @param featureColumn
	 * @param minValue
	 * @param maxValue
	 * @param splits
	 * @return Total number of dimensions
	 */
	public int addDimension(int featureColumn, double minValue, double maxValue, int splits){
		
		// to do
		// Should check whether the specific featureColumn is already added 
		
		this.dimensions.add(featureColumn);
		this.minValues.add(minValue);
		this.maxValues.add(maxValue);
		this.splitSizes.add(splits);
		
		// Modify the previous multipliers
		for (int i = 0; i < this.multiplier.size(); i++) {
			this.multiplier.set(i, this.multiplier.get(i) * splits);
		}
		
		// /* This did not worked due to how object reference works */
		//		for (Integer multiplier : this.multiplier) {
		//			multiplier = multiplier * splits;
		//		}
		
		this.multiplier.add(1);
		return this.dimensions.size();
	}
	
	/**
	 * Add multiple dimensions
	 * @param featureColumns
	 * @param minValues
	 * @param maxValues
	 * @param splits
	 * @return int : Total number of dimensions
	 */
	public int addDimensions(int[] featureColumns, double[] minValues, double[] maxValues, int[] splits){
		for (int i = 0; i < featureColumns.length; i++) {
			addDimension(featureColumns[i], minValues[i], maxValues[i], splits[i]);
		}
		return this.dimensions.size();
	}
	
	/**
	 * Returns to position of the given feature along the selected dimension
	 * @param dimension
	 * @param featureValue
	 * @return int
	 */
	private int getFeatureSplitNo(int dimension, double featureValue){
		return (int)((featureValue - minValues.get(dimension)) * splitSizes.get(dimension) / 
						(maxValues.get(dimension) - minValues.get(dimension))) ;
	}
	
	// Will return the BinNo based on the selected dimensions of the Histogram
	/* (non-Javadoc)
	 * @see lrtest.IHistogramHelper#getBinNo(org.apache.spark.mllib.linalg.Vector)
	 */
	public int getBinNo(Vector v){
		double[] elements = v.toArray();
		int tempValue = 0;
		for (int i = 0; i < dimensions.size(); i++) {
			tempValue += getFeatureSplitNo(i, elements[this.dimensions.get(i)]) * this.multiplier.get(i);
		}
		//		for (int dimension : this.dimensions) {
		//			tempValue += getFeatureSplitNo(dimension, elements[dimension]) * this.multiplier.get(dimension);
		//		}
		return tempValue;
	}
	
	/**
	 * Will reset the fields of the Histogram helper
	 */
	public void clearDimensions(){
		this.dimensions.clear();
		this.minValues.clear();
		this.maxValues.clear();
		this.splitSizes.clear();
		this.multiplier.clear();
	}

	/* (non-Javadoc)
	 * @see lrtest.IHistogramHelper#getParentId(int, int)
	 */
	public int getParentId(int level, int id) {
		int remainder = id;
		int parentId = 0;
		for (int i = 0; i < level; i++) {
			int axispoint = remainder/this.multiplier.get(i);
			remainder -= (axispoint * this.multiplier.get(i));
			parentId += axispoint * this.multiplier.get(i);
		}
		return parentId;
	}

	/* (non-Javadoc)
	 * @see lrtest.IHistogramHelper#getNoofDimensions()
	 */
	public int getNoofDimensions() {
		return this.dimensions.size();
	}

	/* (non-Javadoc)
	 * @see lrtest.IHistogramHelper#getCordinate(int)
	 */
	public int[] getCordinate(int id) {
		int remainder = id;
		int noofDimensions = getNoofDimensions();
		int[] cordinates = new int[noofDimensions];
		for (int i = 0; i < noofDimensions; i++) {
			int axispoint = remainder/this.multiplier.get(i);
			remainder -= (axispoint * this.multiplier.get(i));
			cordinates[i] = axispoint;
		}
		return cordinates;
	}

	/* (non-Javadoc)
	 * @see lrtest.IHistogramHelper#getSplits(int)
	 */
	public int getSplits(int level) {
		return splitSizes.get(level);
	}
}
