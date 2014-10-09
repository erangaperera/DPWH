package lrtest;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.junit.Test;
import org.wso2.carbon.lrtest.HistogramHelper;

import junit.framework.TestCase;

public class TestHistogramHelper extends TestCase {

	@Test
	public void testAddDimension() throws Exception {

		double[] row = { -1, 7, -8, 3, -9 };
		Vector vector = Vectors.dense(row);

		// Test 1
		HistogramHelper histhelper = new HistogramHelper();
		// Add First Dimension
		histhelper.addDimension(0, -50, 50, 10);
		// Add Second Dimension
		histhelper.addDimension(1, 0, 10, 5);
		// Add Third Dimension
		histhelper.addDimension(2, -10, 10, 4);

		int retValue = histhelper.getBinNo(vector);
		assertEquals(92, retValue);
	}

	@Test
	public void testAddDimensions() throws Exception {

		// Test 2
		double[] row = { -1, 7, -8, 3, -9 };
		Vector vector = Vectors.dense(row);

		HistogramHelper histhelper = new HistogramHelper();
		// Add multiple dimensions
		histhelper.addDimensions(new int[] { 0, 1, 2 }, new double[] { -50, 0,
				-10 }, new double[] { 50, 10, 10 }, new int[] { 10, 5, 4 });
		int retValue = histhelper.getBinNo(vector);
		assertEquals(92, retValue);
	}

	@Test
	public void testChangeAxisOrder() throws Exception {

		// Test3, First and Second Column interchanged
		double[] row = { 7, -1, -8, 3, -9 };
		Vector vector = Vectors.dense(row);

		HistogramHelper histhelper = new HistogramHelper();
		// Add Second Column as First Dimension
		histhelper.addDimension(1, -50, 50, 10);
		// Add First Column as Second Dimension
		histhelper.addDimension(0, 0, 10, 5);
		// Add Third Dimension
		histhelper.addDimension(2, -10, 10, 4);

		int retValue = histhelper.getBinNo(vector);
		assertEquals(92, retValue);

	}

}
