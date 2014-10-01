package lrtest;

import java.util.Map;
import java.util.TreeMap;

public class HistogramTree {
	
	private int id;
	private int level;
	private IHistogramHelper histogramHelper;
	private int[] cordinate;
	private int subTotal = 0;
	private final Map<Integer,HistogramTree> children = new TreeMap<Integer,HistogramTree>();
	private boolean leafLevel;
	
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
}
