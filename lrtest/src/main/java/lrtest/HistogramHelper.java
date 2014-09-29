package lrtest;

import java.util.ArrayList;
import org.apache.spark.util.Vector;

@SuppressWarnings("deprecation")
public class HistogramHelper {
	
	private final ArrayList<Integer> dimensions = new ArrayList<Integer>();
	private final ArrayList<Double> minValues = new ArrayList<Double>();
	private final ArrayList<Double> maxValues = new ArrayList<Double>();
	private final ArrayList<Integer> splitSizes = new ArrayList<Integer>();
	private final ArrayList<Integer> multiplier = new ArrayList<Integer>();
	
	// Add one dimension at a time
	public int addDimension(int featureColumn, double minValue, double maxValue, int splits){
		
		// todo
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
	
	// Add multiple dimensions
	public int addDimensions(int[] featureColumns, double[] minValues, double[] maxValues, int[] splits){
		for (int i = 0; i < featureColumns.length; i++) {
			addDimension(featureColumns[i], minValues[i], maxValues[i], splits[i]);
		}
		return this.dimensions.size();
	}
	
	// Returns to position of the given feature along the selected dimension
	private int getFeatureSplitNo(int dimension, double featureValue){
		return (int)((featureValue - minValues.get(dimension)) * splitSizes.get(dimension) / 
						(maxValues.get(dimension) - minValues.get(dimension))) ;
	}
	
	// Will return the BinNo based on the selected dimensions of the Histogram
	public int getBinNo(Vector v){
		double[] elements = v.elements();
		int tempValue = 0;
		for (int dimension : this.dimensions) {
			tempValue += getFeatureSplitNo(dimension, elements[dimension]) * this.multiplier.get(dimension);
		}
		return tempValue;
	}
	
	public void clearDimensions(){
		this.dimensions.clear();
		this.minValues.clear();
		this.maxValues.clear();
		this.splitSizes.clear();
		this.multiplier.clear();
	}
}
