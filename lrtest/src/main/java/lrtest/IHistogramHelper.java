package lrtest;

import java.util.Map;

import org.apache.spark.util.Vector;

@SuppressWarnings("deprecation")
public interface IHistogramHelper {

	// Will return the BinNo based on the selected dimensions of the Histogram
	public abstract int getBinNo(Vector v);

	// Will return the parentId @ given level
	public abstract int getParentId(int level, int id);
	
	// Will return the No of Dimensions that has been added
	public abstract int getNoofDimensions();

	// Create the coordinate for the id
	public abstract int[] getCordinate(int id);

}