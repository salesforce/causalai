import React, {useEffect, useState, createRef} from 'react'
import NodeGraph from '../components/NodeGraph'
import axios from 'axios'

const PerformCausalDiscovery = (props) => {
  if(Object.keys(props).length === 0){
    return <div>no data</div>
  }
  let dataType = props.dataType
  let dataArray = props.dataArray;
  let varNames = props.varNames;
  let causalGraph = props.causalGraph;
  let isDiscrete = props.isDiscrete;
  let originalGraph = props.causalGraph;
  const [var1, setVar1] = useState(null)
  const [var2, setVar2] = useState(null)

  const [priorKnowledgeSection, setPriorKnowledgeSection] = useState(false)
  const [specification, setSpecification] = useState('');
  const [discoveryPerformed, setDiscoveryPerformed] = useState(false);

  let priorKnowledge;
  let priorKnowledgeBuildArray = [];
  const [maxLag, setMaxLag] = useState(null);
  const [algorithm, setAlgorithm] = useState('')
  const [pvalue, setPvalue] = useState(.05)
  const [ciTest, setCiTest] = useState('Partial Correlation')
  const [graphData, setGraphData] = useState([])
  const [precision, setPrecision] = useState(0);
  const [recall, setRecall] = useState(0);
  const [f1Score, setF1Score] = useState(0);

  const [priorKnowledgeArray, setPriorKnowledgeArray] = useState([])
  const [loading, setLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState(false)
  
  const [priorKnowledgeCheck, setPriorKnowledgeCheck] = useState(true)
  const [priorKnowledgeError, setPriorKnowledgeError] = useState('')
  const [varRefs, setVarRefs] = useState([])
  const [undirectedGraph, setUndirectedGraph] = useState(null)

  useEffect(() => {
    let temp = []
    for(let i = 0; i <varNames.length*2; i++){
      temp.push(createRef())
    }
    setVarRefs(temp)
  },[]);

  const deletePriorKnowledge = (index) => {
    let copyArr = [...priorKnowledgeArray];
    copyArr.splice(index, 1)
    setPriorKnowledgeArray(copyArr)
    priorKnowledgeBuildArray = copyArr;
  }
  
  const validate = () => {
    if(maxLag === null && dataType === 'Time Series'){
      return false
    }
    else if(pvalue === null || pvalue > 1){
      return false
    }
    else{
      return true;
    }
  }

  const translateSpecification = (specification) =>{
    if(specification === 'forbidden_links'){
      return 'Forbidden Links'
    }
    else if(specification === 'existing_links'){
      return 'Existing Links'
    }
    else if(specification === 'root_nodes'){
      return 'Root Nodes'
    }
    else if(specification === 'leaf_nodes'){
      return 'Leaf Nodes'
    }
    else{
      return ''
    }
  }
  const handleSpecificationChange = async (e) => {
    setSpecification(e.target.value);
    setVar2(null)
  }
  const handleSetAlgorithm = (value) => {
    setAlgorithm(value)
    if(value === 'VARLINGAM'){
      setPriorKnowledgeArray([]);
      priorKnowledgeBuildArray = [];
      setPriorKnowledgeSection(false);
      setVar1(null)
      setVar2(null)

    }
  }
  const handleAddPriorKnowledge = async () => {

    let tempPrior = JSON.stringify(priorKnowledgeArray)
    let tempMatch
    if(var2 === null){
      tempMatch = JSON.stringify([specification,var1])
    }
    else{
      tempMatch = JSON.stringify([specification,var1,var2])
    }
    let tempExists = tempPrior.indexOf(tempMatch)
    if(tempExists != -1){
    }
    else{
      if(specification === 'forbidden_links' || specification === 'existing_links'){
        if(var1 && var2){
          priorKnowledgeBuildArray = [...priorKnowledgeArray, [specification,var1,var2]]
          await setPriorKnowledgeArray(priorKnowledgeArray => [...priorKnowledgeArray, [specification,var1,var2]])

        }
      }
      else if(specification === ''){

      }
      else{
        if(var1 != null){
          priorKnowledgeBuildArray = [...priorKnowledgeArray, [specification,var1,var2]]
          await setPriorKnowledgeArray(priorKnowledgeArray => [...priorKnowledgeArray, [specification,var1]])
        }
      }
    }
    await handlePriorKnowledgeCheck()
    setVar2(null)
  }

  const handlePriorKnowledgeCheck = () => {
    priorKnowledge = {
      'forbidden_links': {},
      'existing_links': {},
      'root_nodes': [],
      'leaf_nodes': []
    }
    for(let i in priorKnowledgeBuildArray){
      let specification = priorKnowledgeBuildArray[i][0]
      let var1 = priorKnowledgeBuildArray[i][1]
      let var2 = priorKnowledgeBuildArray[i][2]
      if(specification == 'forbidden_links' || specification == 'existing_links'){
        if(!priorKnowledge[specification][var2]){
          priorKnowledge[specification][var2] = []
        }
        priorKnowledge[specification][var2].push(var1)
      }
      else{
        if(var1 != null){
          priorKnowledge[specification].push(var1);
        }
      }
    }


    let data = new FormData();
    data.append('data_type', dataType);
    let jsonPrior = JSON.stringify(priorKnowledge);
    data.append('prior_knowledge', jsonPrior);


    var config = {
      method: 'post',
      url: 'http://127.0.0.1:5000/is_latest_link_valid',
      headers: { 
        'Content-Type': 'application/json'
      },
      data : data
    }
    axios(config)
        .then( function (response) {
          setPriorKnowledgeCheck(response.data.bool)
          setPriorKnowledgeError(response.data.string)

          if(response.data.bool === false){
            let tempArr = priorKnowledgeArray;

            setPriorKnowledgeArray(tempArr)
            priorKnowledgeBuildArray = tempArr
          }
        })
        .catch(function (error) {
          console.log(error);
        });
    }


  const handleCausualDiscovery = () => {
    priorKnowledge = {
      'forbidden_links': {},
      'existing_links': {},
      'root_nodes': [],
      'leaf_nodes': []
    }
    for(let i in priorKnowledgeArray){
      let specification = priorKnowledgeArray[i][0]
      let var1 = priorKnowledgeArray[i][1]
      let var2 = priorKnowledgeArray[i][2]
      if(specification == 'forbidden_links' || specification == 'existing_links'){
        if(!priorKnowledge[specification][var2]){
          priorKnowledge[specification][var2] = []
        }
        priorKnowledge[specification][var2].push(var1)
      }
      else{
        if(var1 != null){
          priorKnowledge[specification].push(var1);
        }
      }
    }
    if(validate()){
      setErrorMessage(false)
      setPriorKnowledgeSection(false)
      let data = new FormData();
      data.append('data_type', dataType);
      let jsonArray = JSON.stringify(dataArray);
      data.append('data_array', jsonArray);
      let jsonVars = JSON.stringify(varNames);
      data.append('var_names', jsonVars);
      let causalArray = JSON.stringify(causalGraph);
      data.append('causal_graph', causalArray);

      let jsonPrior = JSON.stringify(priorKnowledge);
      data.append('prior_knowledge', jsonPrior);
  
      data.append('max_lag', maxLag);
      data.append('algorithm', algorithm);
      data.append('ci_test', ciTest);
      data.append('pvalue', pvalue);
      data.append('isDiscrete', isDiscrete)
      
      setLoading(true)
     
      var config = {
        method: 'post',
        url: 'http://127.0.0.1:5000/perform_causal_discovery',
        headers: { 
          'Content-Type': 'application/json'
        },
        data : data
      }
  
      axios(config)
        .then(function (response) {
          setGraphData(response.data.graph_est);
          setUndirectedGraph(response.data.graph_est_undirected)
          setPrecision(response.data.precision);
          setRecall(response.data.recall);
          setF1Score(response.data.f1_score);
          setDiscoveryPerformed(true)
          setLoading(false)
        })
        .catch(function (error) {
          console.log(error);
        });
    }
    else{
      setErrorMessage(true)
    }
  }

  useEffect(() => {
    if(var1 != null){
      if(specification === 'forbidden_links' || specification === 'existing_links'){
        if(varRefs[varNames.indexOf(var1)].current){
          varRefs[varNames.indexOf(var1)].current.style.color = '#1B96FF'
          varRefs[varNames.indexOf(var1)].current.style.border = '1px solid #1B96FF'
          varRefs[varNames.indexOf(var1)].current.style.backgroundColor = '#EEF4FF'
          varRefs[varNames.indexOf(var1)].current.style.fontWeight = 'bold'
        }

      }
      else{

      }

      for(let i = 0; i < varNames.length; i++){
        if(varNames[i] === var1){

        }
        else{
          if(varRefs[varNames.indexOf(varNames[i])].current){
            varRefs[varNames.indexOf(varNames[i])].current.style.color = 'black'
            varRefs[varNames.indexOf(varNames[i])].current.style.border = '1px solid #C9C9C9'
            varRefs[varNames.indexOf(varNames[i])].current.style.backgroundColor = 'white'
            varRefs[varNames.indexOf(varNames[i])].current.style.fontWeight = 'normal'
          }
        }
      }
    }
    if(specification === 'root_nodes' || specification === 'leaf_nodes'){
      handleAddPriorKnowledge()
      // if(var1 != null){
      //   varRefs[varNames[varNames.indexOf(var1)]].current.style.color = 'black'
      //   varRefs[varNames[varNames.indexOf(var1)]].current.style.border = '1px solid #C9C9C9'
      //   varRefs[varNames[varNames.indexOf(var1)]].current.style.backgroundColor = 'white'
      //   varRefs[varNames[varNames.indexOf(var1)]].current.style.fontWeight = 'normal'
      // }
    }
  }, [var1]);

  useEffect(() => {
    if(var2 === null){

    }
    else{
      handleAddPriorKnowledge()
    }
  }, [var2]);

  useEffect(() => {
    if(var1 != null){
      if(varRefs[varNames.indexOf(var1)].current){
        varRefs[varNames.indexOf(var1)].current.style.color = 'black'
        varRefs[varNames.indexOf(var1)].current.style.border = '1px solid #C9C9C9'
        varRefs[varNames.indexOf(var1)].current.style.backgroundColor = 'white'
        varRefs[varNames.indexOf(var1)].current.style.fontWeight = 'normal'
        setVar1(null)
      }

    }
  }, [specification]);

  return (
    <div>
      <div id="causal-discovery-page">
        <div>
          <div>
          <a onClick={() => props.setActivePage(0)} className='back-link'>&lt; Back</a>
      <label htmlFor="" className="panel-label">Causal Discovery</label>
        <div>
          {algorithm != 'VARLINGAM' && (
            <button className="btn-variant" onClick={() => !priorKnowledgeSection ? setPriorKnowledgeSection(true) : setPriorKnowledgeSection(false)}>Provide Prior Knowledge</button>
          )}
          {priorKnowledgeSection && (
            <div id='prior-knowledge-section'>
              <div className="panel">
                <div id='prior-knowledge-page'>
                  <label htmlFor="specification">Specification</label>
                  <select name="specification" id="specification" defaultValue={''} value={specification} onChange={(e) => handleSpecificationChange(e)}>
                    <option value="" disabled>Choose a specification</option>
                    <option value="forbidden_links">Forbidden Links</option>
                    {dataType != "Time Series" && (
                      <option value="existing_links">Existing Links</option>
                    )}
                    <option value="root_nodes">Root Nodes</option>
                    <option value="leaf_nodes">Leaf Nodes</option>
                  </select>            
                  <div>
                    {specification && (
                      <>
                        <div id='var-name-selects'>
                          <div className="select button-list">
                            <label htmlFor="">Var 1</label>
                            {varNames.map((name, index) => {
                              return <button className='prior-knowledge-button' key={index} ref={varRefs[index]} value={name} onClick={() => setVar1(name)}>{name}</button>
                            })}
                          </div>

                          <div className="select button-list">
                            <label htmlFor="">Var 2</label>
                              {((specification === 'forbidden_links' || specification === 'existing_links') && (
                                varNames.map((name, index) =>{
                                  return <button className='prior-knowledge-button' ref={varRefs[index+varNames.length]} value={name} key={index} onClick={() => setVar2(name)}>{name}</button>
                                })
                              ) )}
                          </div>
          
                        </div>
                      {priorKnowledgeError && (
                          <p id='prior-knowledge-error' className='error-message'>{priorKnowledgeError}</p>
                        )}
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
          </div>
          </div>
          <div className='panel'>
            {(dataType === 'Time Series') && (
              <div>
                <label htmlFor="">Max Lag</label>
                <input type="number" name="test_name" min="0" onKeyDown={(e) => { if (!/[0-9]/.test(e.key)){e.preventDefault}}} onChange={(e) => setMaxLag(e.target.value)} />

              </div>
            )}
            <div>
              <label htmlFor="">Algorithm</label>
              <select name="specification" id="specification" defaultValue={''} onChange={(e) => handleSetAlgorithm(e.target.value)}>
                <option value="" disabled>Choose an algorithm</option>
                <option value="PC">PC</option>
                {((dataType == 'Time Series' && isDiscrete == false)) && (
                  <option value="Granger">Granger</option> 
                )} 
                {((dataType == 'Time Series' && isDiscrete == false)) && (
                  <option value="VARLINGAM">VARLINGAM</option>
                )}
                {((dataType == 'Tabular' && isDiscrete == false)) && (
                  <>
                    <option value="GES">GES</option>
                    <option value="LINGAM">LINGAM</option>
                  </>
                )}

              </select>
            </div>
            {algorithm === 'PC' && (
              <div>
                <label htmlFor="">CI Test</label>
                <select name="" id="" onChange={(e) => setCiTest(e.target.value)}> 
                  {!isDiscrete ? <option value="Partial Correlation">Partial Correlation</option> : <option value="Pearson (discrete variables)">Pearson (Discrete Variables)</option>}
                </select>
              </div>
            )}
            {algorithm && (
              <div>
                <div>
                  <label htmlFor="">p-value Threshold</label>
                  <input type="number" min={0} max={1} step={.01} value={pvalue} onKeyDown={(e) => { if (!/^(0(\.\d+)?|1\.0)$/.test(e.key)){e.preventDefault}}} onChange={(e) => setPvalue(e.target.value)}  />
                </div>
                {errorMessage &&(
                  <p id='error-message'>Please Enter Valid Inputs</p>
                )}
                {pvalue > 0 &&(
                  <button id='perform-discovery-button' className='btn' onClick={() => handleCausualDiscovery()}>Perform Causal Discovery</button>
                )}
                {loading && (
                  <div id="loading">
                    <img src={"/images/loader.gif"} alt="" />
                  </div>
                )}
              </div>

            )}


          </div>
        </div>
        <div id="discovery-output prior-graph-output">
          <div>
            {priorKnowledgeArray.length != 0 && (
              <label htmlFor="" className='panel-label'>Prior Knowledge</label>
            )}
            {(priorKnowledgeArray.length != 0) && (
              <div id={'treatment-vars-section'} className="panel">
                <table>
                  <thead>
                    <tr>
                      <th>Specification</th>
                      <th>Var 1</th>
                      <th>&rarr;</th>
                      <th>Var 2</th>
                    </tr>
                    <tr className='spacer'></tr>
                  </thead>
                  <tbody>
                    {priorKnowledgeArray.map((item, index) => {
                      return(
                        <React.Fragment key={index}>
                        <tr className='data-row' key={index}>
                          <td>{translateSpecification(item[0])}</td>
                          <td>{item[1]}</td>
                          <td>{item[0] === 'forbidden_links' || item[0] === 'existing_links' ? <>&rarr;</> : <></> }</td>
                          <td>{item[2]}</td>
                          <div className="button-cell"><a className='' onClick={() => deletePriorKnowledge(index)}>Remove</a></div>
                        </tr>
                        <tr className='spacer'></tr>
                        </React.Fragment>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
          
        </div>
        <div id='discovery-graphs'>
          {Object.keys(originalGraph).length != 0 && (
            <div id="discovery-outputs">
              <label htmlFor="" className='panel-label'>Ground Truth Graph</label>
              <NodeGraph data={originalGraph} dataType={dataType} graphId={'cy'}/>
            </div>
          )}
      
          {(discoveryPerformed && !loading) && (
            <div id="discovery-outputs">
              <label htmlFor="" className="panel-label">Estimated Graph</label>

              <NodeGraph data={graphData} dataType={dataType} graphId={'cy2'} />
              <div className="discovery-data">
                {precision != null && (
                  <>
                    <h4>Precision: {parseFloat(precision.toFixed(2))}</h4>
                    <h4>Recall : {parseFloat(recall.toFixed(2))}</h4>
                    <h4>f1 Score: {parseFloat(f1Score.toFixed(2))}</h4>
                  </>
                )}      
              </div>
              {discoveryPerformed && (
                <div id='download-graph-btns'>
                  <a 
                    type='button' 
                    id='download-graph-button' 
                    className='btn'
                    href={`data:text/json;charset=utf-8,${encodeURIComponent(
                      JSON.stringify(graphData)
                    )}`}
                    download="causal-discovery-graph.json"
                  >
                  {dataType == 'Tabular' ? 'Download Oriented Graph' : 'Download Graph'}
                  </a>
                  {dataType == "Tabular" && (
                    <a 
                      type='button' 
                      id='download-graph-button' 
                      className='btn'
                      href={`data:text/json;charset=utf-8,${encodeURIComponent(
                        JSON.stringify(undirectedGraph)
                      )}`}
                      download="causal-discovery-undirected-graph.json"
                    >
                    Download Un-Oriented Graph
                    </a>
                  )}
            
                </div>
                
            )}
            </div>
          )}
          </div>
      </div>

    </div>
  )
}

export default PerformCausalDiscovery