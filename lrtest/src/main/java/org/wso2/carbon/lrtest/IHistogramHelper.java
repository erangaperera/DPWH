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

//import org.apache.spark.util.Vector;
import org.apache.spark.mllib.linalg.Vector;

/**
 * Define the contract that a HistogramHelper should implement
 * @author erangap@wso2.com
 */
public interface IHistogramHelper {

	// Will return the BinNo based on the selected dimensions of the Histogram
	public abstract int getBinNo(Vector v);

	// Will return the parentId @ given level
	public abstract int getParentId(int level, int id);
	
	// Will return the No of Dimensions that has been added
	public abstract int getNoofDimensions();

	// Create the coordinate for the id
	public abstract int[] getCordinate(int id);

	// Returns the number of splits on the given dimension
	public abstract int getSplits(int level);

}