import React, {useEffect, useState} from 'react';
import cytoscape from "cytoscape";
import dagre from "cytoscape-dagre";
import cola from "cytoscape-cola";
import avsdf from "cytoscape-avsdf";
import cise from "cytoscape-cise";
import coseBilkent from "cytoscape-cose-bilkent";
import spread from 'cytoscape-spread'



const NodeGraph = (props) => {

  if(Object.keys(props).length === 0){
    return <div>no data</div>
  }
  let nodesArray = []
  let edgeArray = []
  let data = props.data
  let lowestNum = 0;
  let nodes = []
  let edges = []
  let layout = ''
  let rows = 0
  let graphId = props.graphId;
  if(data){
    rows = Object.keys(data).length
  }

  if(props.dataType === 'Time Series'){
    for(const arr in data){
      nodesArray.push(arr)
      for(const dataArr in data[arr]){
        edgeArray.push(arr + ' ' + data[arr][dataArr][0] + '(t' + data[arr][dataArr][1] + ')')
        if(data[arr][dataArr][1] < lowestNum){
          lowestNum = data[arr][dataArr][1]
        }
      }
    }

    for(const node in nodesArray){
      for(let i = Math.abs(lowestNum); i >= 0; --i){
        if(i == 0){
          nodes.push({data: {id:  String(nodesArray[node]), name: nodesArray[node] + '(t)'}})
        }
        else{
          nodes.push({data: {id: nodesArray[node] + '(t-' + i + ')' , name: nodesArray[node] + '(t-' + i + ')' }})
        }
      }
    }
    for(const edge in edgeArray){
      edges.push({data: {id:edgeArray[edge], source:edgeArray[edge].split(' ')[1], target: edgeArray[edge].split(' ')[0], value: 1}})
    }

    layout={
      name: 'grid',
      rows: rows,
      infinite: true,
    }
  }
  if(props.dataType === 'Tabular'){
    for(const arr in data){
      nodesArray.push(arr)
      for(const dataArr in data[arr]){
        edgeArray.push(arr + ' ' + data[arr][dataArr])
      }
    }
    for(const node in nodesArray){
      nodes.push({data: {id:  String(nodesArray[node]), name: nodesArray[node]}})
    }
    for(const edge in edgeArray){
      edges.push({data: {id:edgeArray[edge], source:edgeArray[edge].split(' ')[1], target: edgeArray[edge].split(' ')[0], value: 1}})
    }
    layout={
      name: 'cola',
    }
  }

  useEffect(() => {
    
    const elements = {
      nodes: nodes,
      edges: edges
      
    };
    cytoscape.use(dagre)
    cytoscape.use(avsdf)
    cytoscape.use(cise)
    cytoscape.use(coseBilkent)
    cytoscape.use(spread)
    cytoscape.use(cola)
    var cy = (window.cy = cytoscape({
      headless:false,

      container: document.getElementById(graphId),
    
      boxSelectionEnabled: false,
      autounselectify: true,
    
      // layout: layout,
      zoomingEnabled:false,
      style: [
        {
          selector: "node",
          style: {
            'height': 20,
            'width': 20,
            'backgroundColor': '#014486',
            'opacity': 1,
            'label': 'data(name)'
          }
        },
        {
          selector: "edge",
          style: {
            'curve-style': 'bezier',
            'haystack-radius': 1,
            'width': 5,
            'line-color': '#1AB9FF',
            'opacity': 1,
            'target-arrow-shape': 'triangle',
            'target-arrow-color': '#1AB9FF '
            

          }
        }
      ],
    
      elements
    }));
    if(props.dataType === 'Tabular'){
      cy.layout({
        name: 'cola',
        randomize:true,
        // centerGraph:false,
      }).run()  
    }
    else{
      cy.layout({
        name: 'grid',
        rows: rows,
        infinite: true,
      }).run()
    }

  },[])


  return (
    <div id={graphId}></div>   
  );
}

export default NodeGraph