package lrtest;

import org.apache.spark.util.Vector;

@SuppressWarnings("deprecation")
public class VectorTest {

	public static void main(String[] args) {
		
		// Test Data
		double[] row = { -1, 7, -8, 3, -9 };
		Vector v1 = new Vector(row);
		
		// Test 1
		HistogramHelper histhelper = new HistogramHelper();
		// Add First Dimension
		histhelper.addDimension(0, -50, 50, 10);
		// Add Second Dimension
		histhelper.addDimension(1, 0, 10, 5);
		// Add Third Dimension
		histhelper.addDimension(2, -10, 10, 4);
		
		
		int retValue = histhelper.getBinNo(v1);
		// Should print 92
		System.out.println(retValue);
		
		// Test 2
		HistogramHelper histhelper2 = new HistogramHelper();
		// Add multiple dimensions
		histhelper2.addDimensions(new int[] {0, 1, 2}, 
								  new double[] {-50, 0, -10}, 
								  new double[] {50, 10, 10}, 
								  new int[]{10, 5, 4});
		retValue = histhelper2.getBinNo(v1);
		// Should Print 92
		System.out.println(retValue);
		
		// Test 3
		// Should work
		histhelper.clearDimensions();
		histhelper.addDimensions(new int[] {0, 1, 2}, 
				  new double[] {-50, 0, -10}, 
				  new double[] {50, 10, 10}, 
				  new int[]{10, 5, 4});
		retValue = histhelper.getBinNo(v1);
		// Should Print 92
		System.out.println(retValue);
		
		
		System.out.println(System.currentTimeMillis());
		//repeat 1 million times
		for (int i = 0; i < 100000000; i++) {
			retValue = histhelper.getBinNo(v1);
		}
		System.out.println(System.currentTimeMillis());
	}

}
