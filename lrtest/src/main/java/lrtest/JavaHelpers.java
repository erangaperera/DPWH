package lrtest;

import org.apache.spark.util.Vector;

public class JavaHelpers {

	public static String getSparkUrl() {
		// TODO Auto-generated method stub
		return null;
	}

	public static String getMasterHostname() {
		// TODO Auto-generated method stub
		return null;
	}

	@SuppressWarnings("deprecation")
	public static Vector parseVector(String line) {
		
		String[] parts = line.split(",");
		double[] vElms = new double[parts.length];
		for (int i = 0; i < vElms.length; i++) {
			if ( parts[i] == null || parts[i].isEmpty()){
				vElms[i] = 0.0;
            }else{
            	vElms[i] = Double.parseDouble(parts[i]);
            }
		}
		return new Vector(vElms);
	}

}
