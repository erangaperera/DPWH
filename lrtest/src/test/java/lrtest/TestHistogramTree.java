package lrtest;

import static org.junit.Assert.*;

import java.util.Map;

import org.apache.spark.util.Vector;
import org.junit.Test;
import org.wso2.carbon.lrtest.HistogramHelper;
import org.wso2.carbon.lrtest.HistogramTree;

@SuppressWarnings("deprecation")
public class TestHistogramTree {

	@Test
	public void testGroupsofEqualSize() {

		// Create sample bins
		int size = 8 * 8 * 8; // Dimensions - 8 x 8 x 8
		int[] bins = new int[size];
		for (int i = 0; i < 10000; i++) {
			int frequency = (int) (Math.random() * 3);
			bins[(int) (Math.random() * size)] += frequency;
		}

		HistogramHelper histhelper = new HistogramHelper();
		// Add multiple dimensions
		histhelper.addDimensions(new int[] { 0, 1, 2 }, new double[] {
		Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY,
		Double.NEGATIVE_INFINITY }, new double[] {
		Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY,
		Double.POSITIVE_INFINITY }, new int[] { 8, 8, 8 });
		
		HistogramTree tree = new HistogramTree(histhelper);
		int totFreq = 0;
		for (int i = 0; i < bins.length; i++) {
			tree.AddNode(i, bins[i]);
			totFreq += bins[i];
		}
		
		Map<Integer, Integer[]> bingroups = tree.groupBins(8);
		double avg = totFreq / bingroups.size();
		// Expected Group size +/- 10%
		double expectedMin = avg * 0.9;
		double expectedMax = avg * 1.1;
		for (int i = 0; i < bingroups.size(); i++) {
			Integer[] groups = bingroups.get(i);
			double groupTot = 0.0;
			for (Integer binId : groups) {
				groupTot += bins[binId];
			}
			// System.out.println("Total -> " + groupTot + ", Min -> "+ expectedMin + ", Max -> " + expectedMax);
			assertTrue("Bin Group total is lower than expected ", (groupTot > expectedMin));
			assertTrue("Bin Group total is higher than expected ", (groupTot < expectedMax));
		}

	}

	@Test
	public void testGroupMembersAdjacent() {

		HistogramHelper histhelper = new HistogramHelper();
		
		// Add multiple dimensions
		histhelper.addDimensions(new int[] { 0, 1, 2 }, new double[] {
				Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY,
				Double.NEGATIVE_INFINITY }, new double[] {
				Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY,
				Double.POSITIVE_INFINITY }, new int[] { 8, 8, 8 });
		
		HistogramTree tree = new HistogramTree(histhelper);

		// Populate the HistogramTree
		int b = 512;
		for (int i = 0; i < 10000; i++) {
			int a = (int) (Math.random() * 3);
			tree.AddNode((int) (Math.random() * b), a);
		}

		// Test the grouping
		Map<Integer, Integer[]> bingroups = tree.groupBins(8);
		for (int i = 0; i < bingroups.size(); i++) {
			// Loop through the group
			Integer[] binGroup = bingroups.get(i);
			// All the members in the group should be a direct adjacent of at
			// least of one another
			for (int j = 0; j < binGroup.length; j++) {
				double minDistance = Double.POSITIVE_INFINITY;
				Vector firstBin = getVector(histhelper
						.getCordinate(binGroup[j]));
				for (int k = 0; k < binGroup.length; k++) {
					Vector secondBin;
					if (j == k)
						continue;
					else
						secondBin = getVector(histhelper
								.getCordinate(binGroup[k]));
					if (firstBin.squaredDist(secondBin) < minDistance)
						minDistance = firstBin.squaredDist(secondBin);
				}
				assertEquals(1.0, minDistance, 0.0000001);
			}
		}
	}

	// Get the vector of given coordinate array, Vector has inbuilt support for calculations
	private static Vector getVector(int[] cordinate) {
		double cordinateArray[] = new double[cordinate.length];
		for (int i = 0; i < cordinate.length; i++) {
			cordinateArray[i] = cordinate[i];
		}
		return new Vector(cordinateArray);
	}

}
