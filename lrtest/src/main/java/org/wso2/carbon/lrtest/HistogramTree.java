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
import java.util.Map;
import java.util.TreeMap;

import org.apache.log4j.Logger;

/**
 * This class will hold the structure of a histogram
 * @author erangap
 */
public class HistogramTree implements Serializable{
	
	private static final long serialVersionUID = -2753020274537678149L;
	private int id;
	private int level;
	private IHistogramHelper histogramHelper;
	private int[] cordinate;
	private int subTotal = 0;
	private final Map<Integer,HistogramTree> children = new TreeMap<Integer,HistogramTree>();
	private boolean leafLevel;
	private int groupId = 0;
	
	public int getId() {
		return id;
	}

	public int[] getCordinate() {
		return cordinate;
	}

	public int getSubTotal() {
		return subTotal;
	}

	// Public constructor to create the Root Node
	public HistogramTree(IHistogramHelper histogramHelper) {
		this.level = 0;
		this.leafLevel = false;
		this.histogramHelper = histogramHelper;
	}
	
	// This will only Allow node addition through the Root.
	private HistogramTree(IHistogramHelper histogramHelper, int level, int id) {
	
		this.id = id;
		this.level = level;
		this.histogramHelper = histogramHelper;
		this.cordinate = histogramHelper.getCordinate(id);
		this.leafLevel = (histogramHelper.getNoofDimensions() == level);
	}

	public Map<Integer, HistogramTree> getChildren() {
		return children;
	}

	// This will create / Update relevant child nodes 
	public void AddNode(int id, int subTotal){
		
		// Update sub-totals
		this.subTotal += subTotal;
		
		// If We are at the leaf level not required to cascade the updates
		if (leafLevel)
			return;
		
		// We are at intermediate level, Needs to do update the subtrees
		int nextLevel = this.level + 1;
		int parentId = this.histogramHelper.getParentId(nextLevel, id);
		if (this.children.containsKey(parentId)){
			HistogramTree parent = this.children.get(parentId);
			
			// parent.subTotal += subTotal;
			parent.AddNode(id, subTotal);
		}
		else
		{
			HistogramTree parent = new HistogramTree(this.histogramHelper, nextLevel, parentId);
			parent.AddNode(id, subTotal);
			this.children.put(parentId, parent);
		}
	}

	// This will group the adjacent bins of roughly equal size
	public Map<Integer, Integer[]> groupBins(int noofBins) {
		int noofDimensions = histogramHelper.getNoofDimensions();
		double divideFactor = Math.pow(noofBins, (1.0 / noofDimensions));
		Logger.getRootLogger().debug("Total number of Data Points -> " + this.subTotal);
		Logger.getRootLogger().debug("Required No. of bins -> " + noofBins);
		return groupBins(children, 0, this.subTotal, divideFactor);
	}
	
	// We will use recursion to navigate deep into the tree
	private Map<Integer, Integer[]> groupBins(Map<Integer,HistogramTree> subtree,int level, double subTotal, double divideFactor) {
		
		double qty2split = subTotal / divideFactor;
		int dimensionSize = histogramHelper.getSplits(level);
		
		// This will aggregate the results received from the recursion
		Map<Integer, Integer[]> binGroup = new TreeMap<Integer, Integer[]>();
		// This variable will combine the child nodes that will send to the next level
		Map<Integer,HistogramTree> nextTree = new TreeMap<Integer, HistogramTree>();
		int splitTotal = 0;
		
		for (int i = 0; i < dimensionSize; i++) {
			
			int loopTotal = 0;
			int nextLoopTotal = 0;
			
			Map<Integer,HistogramTree> tempTree = new TreeMap<Integer, HistogramTree>();
			Map<Integer, HistogramTree> removeTree = new TreeMap<Integer, HistogramTree>();
			
			for (HistogramTree node : subtree.values()) {
				if (node.getCordinate()[level] == i){
					loopTotal += node.getSubTotal();
					// tempTree.put(node.getId(), node);
					removeTree.put(node.getId(), node);
					// if we are in the last level add current nodes else add children
					if (level == (histogramHelper.getNoofDimensions() - 1)){
						tempTree.put(node.getId(), node);
					}
					else
					{
						tempTree.putAll(node.getChildren());
					}
				}
				nextLoopTotal += (node.getCordinate()[level] == i + 1) ? node.getSubTotal() : 0;
			}
			splitTotal += loopTotal;
			nextTree.putAll(tempTree);
			
			// Needs some improvement Here
			if ((splitTotal > qty2split) || (((splitTotal <= qty2split) && (qty2split < (splitTotal + nextLoopTotal))) && ((qty2split - splitTotal) < (splitTotal + nextLoopTotal - qty2split)))){
				// Make a comparison of for next start value
				if (level == (histogramHelper.getNoofDimensions() - 1)){
					binGroup.putAll(createGroup(nextTree));
				}
				else{
					binGroup.putAll(groupBins(nextTree, level +1, qty2split, divideFactor));
				}
				// Current sub-total is sent for the next level to iterate, therefore clear
				nextTree.clear();
				splitTotal = 0;
			}
			else{
				for (HistogramTree node : removeTree.values()) {
					subtree.remove(node.getId());
				}
			}
		}
		
		// This is the last group along the axis
		if (splitTotal > 0){
			if (level == (histogramHelper.getNoofDimensions() - 1)){
				binGroup.putAll(createGroup(nextTree));
			}
			else{
				binGroup.putAll(groupBins(nextTree, level +1, qty2split, divideFactor));
			}
		}
		
		return binGroup;
	}
	
	private Map<Integer, Integer[]> createGroup(Map<Integer,HistogramTree> tree){
		
		Map<Integer, Integer[]> temp = new TreeMap<Integer, Integer[]>();
		Integer[] bins = new Integer[tree.size()];
		int i = 0;
		int groupTotal = 0;
		for(HistogramTree node : tree.values()){
			bins[i] = node.getId();
			groupTotal += node.getSubTotal();
			i++;
		}
		temp.put(groupId, bins);
		
		Logger.getRootLogger().debug("Group created -> " + groupId + ", Total -> " + groupTotal);
		groupId++;
		return temp;
	}
}
