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

import java.util.List;

public class Metrics {

	public static double accuracy(double[] actual, double[] predicted) {

		assert (actual.length == predicted.length);

		double accuracy = 0.0;
		for (int i = 0; i < actual.length; i++) {
			if (predicted[i] == actual[i]) {
				accuracy++;
			}
		}

		return (accuracy / actual.length) * 100.0;
	}

	public static double accuracy(List<Double> actual, List<Double> predicted) {

		assert (actual.size() == predicted.size());

		double accuracy = 0.0;
		for (int i = 0; i < actual.size(); i++) {
			if (predicted.get(i).doubleValue() == actual.get(i).doubleValue()) {
				accuracy++;
			}
		}

		return (accuracy / actual.size()) * 100.0;
	}

}
