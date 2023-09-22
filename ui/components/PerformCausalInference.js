import React, {useEffect, useState} from 'react'
import axios from 'axios'
import NodeGraph from '../components/NodeGraph'



const PerformCausalInference = (props) => {
  if(Object.keys(props).length === 0){
    return <div>no data</div>
  }
  let dataType = props.dataType
  let dataArray = props.data;
  let varNames = props.varNames;
  let causalGraph = props.causalGraph;
  let isDiscrete = props.isDiscrete;
  let randomSeed = props.randomSeed;
  let isDataGenerated = props.isDataGenerated;
  let numVars = props.numVars;
  let numSamples = props.numSamples;
  let maxLag = props.maxLag;
  let undirectedEdgesArray = props.undirectedEdges;

  const [graphState, setGraphState] = useState(causalGraph)
  const [addedVariable, setAddedVariable] = useState(false);
  const [estimatedAte, setEstimatedAte] = useState(false);
  const [estimatedCate, setEstimatedCate] = useState(false);
  const [estimatedCounterfactual, setEstimatedCounterfactual] = useState(false);
  const [addedConditionVariable, setAddedConditionVariable] = useState(false)

  const [targetVar, setTargetVar] = useState('')

  const [predictionModel, setPredictionModel] = useState('')
  const [treatmentVar, setTreatmentVar] = useState('');
  const [treatmentValue, setTreatmentValue] = useState(0)
  const [controlValue, setControlValue] = useState(0)

  const [conditionVar, setConditionVar] = useState('')
  const [conditionValue, setConditionValue] = useState(0)
  const [undirectedEdges, setUndirectedEdges] = useState(undirectedEdgesArray)

  const [selectedEdge, setSelectedEdge] = useState('')

  const [treatmentVars, setTreatmentVars] = useState([])
  const [conditionVars, setConditionVars] = useState([])

  const [estAte, setEstAte] = useState(0)
  const [trueAte, setTrueAte] = useState(0)
  const [estCate, setEstCate] = useState(0)
  const [trueCate, setTrueCate] = useState(0)
  const [estCounterfactual, setEstCounterfactual] = useState(0)
  const [trueCounterfactual, setTrueCounterfactual] = useState(0)

  const [loadingAte, setLoadingAte] = useState(true)
  const [loadingCate, setLoadingCate] = useState(true)
  const [loadingCounterfactual, setLoadingCounterfactual] = useState(true)

  const [undirectedEdgesSection, setUndirectedEdgesSection] = useState(false)
  const [undirectedDone, setUndirectedDone] = useState(false)
  const [toggleUndirectedGraph, setToggleUndirectedGraph] = useState(true)
  const [maxStateVals, setMaxStateVals] = useState({})

  const [treatmentError, setTreatmentError] = useState('')
  const [conditionError, setConditionError] = useState('')
  const [graph, setGraph] = useState(null)

  const handleAddTreatmentVars = () => {
    if(isDiscrete){
      if(maxStateVals[treatmentVar] < treatmentValue){
        setTreatmentError('Enter a valid treatment value')
        return;
      }
      else if(maxStateVals[treatmentVar] < controlValue){
        setTreatmentError('Enter a valid control value')
        return;
      }
    }

    setTreatmentError('')
    for(let i = 0; i < treatmentVars.length; i++){
      if(treatmentVar === treatmentVars[i][0]){
        deleteTreatmentVar(i);
      }
    }
    if(treatmentValue != null && controlValue != null){
      
      setTreatmentVars(treatmentVars => [...treatmentVars, [treatmentVar,parseFloat(treatmentValue),parseFloat(controlValue)]])
    }
  }
  const handleAddConditionVars = () => {
    if(isDiscrete){
      if(maxStateVals[conditionVar] < conditionValue){
        setConditionError('Enter a valid condition value')
        return;
      }  
    }

    setConditionError('')

    for(let i = 0; i < conditionVars.length; i++){
      if(conditionVar === conditionVars[i][0]){
        deleteConditionVar(i);
      }
    }
    if(conditionValue != null){
      setConditionVars(conditionVars => [...conditionVars, [conditionVar,parseFloat(conditionValue)]])

    }
   }



  const selectLeftEdge = async (edge) => {
    setToggleUndirectedGraph(false)
    edge = edge.split('-');
    edge = edge[0].split(' ')
    edge = edge[0].split(',')
    let copyGraph = causalGraph
    const index = copyGraph[edge[1]].indexOf(edge[0]);
    copyGraph[edge[1]].splice(index, 1)
    causalGraph = copyGraph
    await setGraphState(copyGraph)
    // await setGraph(<NodeGraph data={copyGraph} dataType={dataType} graphId={'cy2'}  />)

    let temp = []
    temp.push(edge[0], edge[1]);
    let copyUndirectedEdges = undirectedEdges.slice();
    let undirectedIndex  
    for(edge in copyUndirectedEdges){
      if((copyUndirectedEdges[edge[0]][0] == temp[0]) && (copyUndirectedEdges[edge[0]][1] == temp[1])){
        undirectedIndex = edge
      }
    }
    copyUndirectedEdges.splice(undirectedIndex, 1)
    // undirectedEdgesArray = copyUndirectedEdges
    await setUndirectedEdges(copyUndirectedEdges)
    if(undirectedEdges.length === 0){
      await setUndirectedDone(true)
    }
    setToggleUndirectedGraph(true)
  }

  const selectRightEdge = async (edge) => {
    setToggleUndirectedGraph(false)
    edge = edge.split('-');
    edge = edge[0].split(' ')
    edge = edge[0].split(',')
    let copyGraph = causalGraph
    const index = copyGraph[edge[0]].indexOf(edge[1]);
    copyGraph[edge[0]].splice(index, 1)
    causalGraph = copyGraph
    await setGraphState(copyGraph)
    // await setGraph(<NodeGraph data={copyGraph} dataType={dataType} graphId={'cy2'}  />)

    let temp = []
    temp.push(edge[0], edge[1]);
    let copyUndirectedEdges = undirectedEdges.slice();
    let undirectedIndex  
    for(edge in copyUndirectedEdges){
      if((copyUndirectedEdges[edge[0]][0] == temp[0]) && (copyUndirectedEdges[edge[0]][1] == temp[1])){
        undirectedIndex = edge
      }
    }
    copyUndirectedEdges.splice(undirectedIndex, 1)
    // undirectedEdgesArray = copyUndirectedEdges
    await setUndirectedEdges(copyUndirectedEdges)
    if(undirectedEdges.length === 0){
      await setUndirectedDone(true)
    }
    setToggleUndirectedGraph(true)
  }
  


  const deleteTreatmentVar = (index) => {
    let copyArr = [...treatmentVars];
    copyArr.splice(index, 1)
    setTreatmentVars(copyArr)
  }
  const deleteConditionVar = (index) => {
    let copyArr = [...conditionVars];
    copyArr.splice(index, 1)
    setConditionVars(copyArr)
  }

  const ateConditionsMet = () =>{
    if(treatmentVars.length > 0){
      return true;
    }
    else{
      return false;
    }
  }

  const cateConditionsMet = () => {
    if(treatmentVars.length > 0 && conditionVars.length > 0){
      return true;
    }
    else{
      return false;
    }
  }

  const counterfactualConditionsMet = () =>{
    let set = new Set()
    if(treatmentVars.length === 1  ){
      set.add(treatmentVars[0][0])
      set.add(targetVar)
      for(let i = 0; i < conditionVars.length; i++){
        set.add(conditionVars[i][0])

      }
      if(set.size === varNames.length){
        
        return true;
      }
      else{
        return false;
      }
    }
    else{
      return false;
    }
  }
  const handleAte = () => {
    setLoadingAte(true)
    setEstimatedCate(false)
    setEstimatedCounterfactual(false)
    let data = new FormData();
    let jsonArray = JSON.stringify(dataArray);
    let jsonVars = JSON.stringify(varNames);
    let causalArray = JSON.stringify(causalGraph);
    let treatmentVarsArray = JSON.stringify(treatmentVars);

    // if(JSON.stringify(causalGraph) === '{}'){
    //   causalArray = JSON.stringify(uploadedCausalGraph);
    // }

		data.append('data_type', dataType);
    data.append('data_array', jsonArray);
		data.append('var_names', jsonVars);

		data.append('causal_graph', causalArray);

    data.append('target_var', targetVar);
    data.append('prediction_model', predictionModel);
    data.append('treatments', treatmentVarsArray)

    data.append('isDiscrete', isDiscrete);
    data.append('random_seed', randomSeed);
    data.append('isDataGenerated', isDataGenerated);
    
    data.append('num_vars', numVars);
    data.append('num_samples', numSamples);
    data.append('max_lag', maxLag);
    setEstimatedAte(true)

    

    var config = {
			method: 'post',
			url: 'http://127.0.0.1:5000/ate',
			headers: { 
				'Content-Type': 'application/json'
			},
			data : data
		}

    axios(config)
      .then(function (response) {
        setEstAte(response.data.est_ate)
        setTrueAte(response.data.true_ate)
        setLoadingAte(false);
      })
      .catch(function (error) {
        console.log(error);
      });
  }
  const handleCate = () => {
    setLoadingCate(true);

    setEstimatedAte(false)
    setEstimatedCounterfactual(false)
		let data = new FormData();
    let jsonArray = JSON.stringify(dataArray);
    let jsonVars = JSON.stringify(varNames);
    let causalArray = JSON.stringify(causalGraph);
    let treatmentVarsArray = JSON.stringify(treatmentVars);
    let conditionVarsArray = JSON.stringify(conditionVars);

    // if(JSON.stringify(causalGraph) === '{}'){
    //   causalArray = JSON.stringify(uploadedCausalGraph);
    // }

		data.append('data_type', dataType);
    data.append('data_array', jsonArray);
		data.append('var_names', jsonVars);
		data.append('causal_graph', causalArray);

    data.append('target_var', targetVar);
    data.append('prediction_model', predictionModel);
    data.append('treatments', treatmentVarsArray)

    data.append('conditions', conditionVarsArray)
    data.append('condition_prediction_model', predictionModel)

    data.append('isDiscrete', isDiscrete);
    data.append('random_seed', randomSeed);
    data.append('isDataGenerated', isDataGenerated);
    
    data.append('num_vars', numVars);
    data.append('num_samples', numSamples);
    data.append('max_lag', maxLag);
    
    setEstimatedCate(true)

    var config = {
			method: 'post',
			url: 'http://127.0.0.1:5000/cate',
			headers: { 
				'Content-Type': 'application/json'
			},
			data : data
		}

    axios(config)
      .then(function (response) {
        setEstCate(response.data.est_cate)
        setTrueCate(response.data.true_cate)
        setLoadingCate(false)
      })
      .catch(function (error) {
        console.log(error);
      });
  }
  const handleCounterfactual = () => {
    setLoadingCounterfactual(true)
    setEstimatedCate(false);
    setEstimatedAte(false)
    let data = new FormData();
    let jsonArray = JSON.stringify(dataArray);
    let jsonVars = JSON.stringify(varNames);
    let causalArray = JSON.stringify(causalGraph);
    let treatmentVarsArray = JSON.stringify(treatmentVars);
    let conditionVarsArray = JSON.stringify(conditionVars);

		data.append('data_type', dataType);
    data.append('data_array', jsonArray);
		data.append('var_names', jsonVars);
		data.append('causal_graph', causalArray);

    data.append('target_var', targetVar);
    data.append('prediction_model', predictionModel);
    data.append('treatments', treatmentVarsArray)

    data.append('conditions', conditionVarsArray)
    data.append('condition_prediction_model', predictionModel)

    data.append('isDiscrete', isDiscrete);
    data.append('random_seed', randomSeed);
    data.append('isDataGenerated', isDataGenerated);
    
    data.append('num_vars', numVars);
    data.append('num_samples', numSamples);
    data.append('max_lag', maxLag);
    setEstimatedCounterfactual(true)

    var config = {
			method: 'post',
			url: 'http://127.0.0.1:5000/counterfactual',
			headers: { 
				'Content-Type': 'application/json'
			},
			data : data
		}

    axios(config)
      .then(function (response) {
        console.log(response)
        setEstCounterfactual(response.data.est_counterfactual)
        setTrueCounterfactual(response.data.true_counterfactual)
        setLoadingCounterfactual(false)
      })
      .catch(function (error) {
        console.log(error);
      });
  }
  useEffect(() => {
		let data = new FormData();
    let jsonArray = JSON.stringify(dataArray);
    let jsonVars = JSON.stringify(varNames);
    data.append('data_array', jsonArray);
		data.append('var_names', jsonVars);

    var config = {
			method: 'post',
			url: 'http://127.0.0.1:5000/find_discrete_data_max_state',
			headers: { 
				'Content-Type': 'application/json'
			},
			data : data
		}

    axios(config)
      .then(function (response) {
        setMaxStateVals(response.data.max_state_val)
      })
      .catch(function (error) {
        console.log(error);
      });
  },[]);


  return (
    <div>
      <a onClick={() => props.setActivePage(0)} className='back-link'>&lt; Back</a>
      {undirectedEdges.length != 0 && (
        <div id='undirected-edges-screen'> 
          <div>        
            <label htmlFor="" className="panel-label">Undirected Edges</label>
            <div className='panel'>
              <select id="multi-select" multiple onChange={(e) => setSelectedEdge(e.target.value)}>
                {undirectedEdges.length != 0  && (
                  undirectedEdges.map((edge,index) =>{
                    return <option className='undirected-options' value={edge} key={index} onClick={(e) => setSelectedEdge(e.target.value) }>{edge[0] + ' - ' + edge[1]}</option>
                  })
                )}
              </select>
              <div className='arrow-btns'>
                <button className="btn" onClick={() => selectLeftEdge(selectedEdge)}>←</button>
                <button className="btn" onClick={() => selectRightEdge(selectedEdge)}>→</button>

              </div>
            </div>
          </div>
          <div>
            {toggleUndirectedGraph && (
              <NodeGraph data={graphState} dataType={dataType} graphId={'cy2'}  />
            )}
          </div>
        </div>

      )}
      {((isDataGenerated || undirectedEdgesArray.length === 0 || undirectedEdges.length === 0) ) && (
        <div id='causal-inference'>
          <div>
            <label htmlFor="" className="panel-label" id='condition-label'>Causal Inference</label>
            <NodeGraph data={causalGraph} dataType={dataType} graphId={'cy'}  />
            <div id='inference-buttons'>
              <div className='flex'>
                <button disabled={ateConditionsMet() ? false : true} className="btn" onClick={() => handleAte()}>Estimate ATE</button>
                <div className='flex-column'>
                  <span className='required'>Required:</span>
                  <span>{treatmentVars.length > 0 ? (<img src='/images/check.svg' />) :  (<img src='/images/close-2.svg' />)}At least one treatment variable</span>
                </div>
              </div>
              <div className='flex'>
                <button disabled={cateConditionsMet() ? false : true} className="btn"onClick={() => handleCate()}>Estimate CATE</button>
                <div className='flex-column'>
                  <span className='required'>Required:</span>
                  <span>{treatmentVars.length > 0 ? (<img src='/images/check.svg' />) :  (<img src='/images/close-2.svg' />)} At least one treatment variable</span>
                  <span>{conditionVars.length > 0 ? (<img src='/images/check.svg' />) :  (<img src='/images/close-2.svg' />)} At least one condition variable</span>
                </div>
              </div>
              <div className='flex'>
                <button disabled={counterfactualConditionsMet() ? false : true} className="btn"onClick={() => handleCounterfactual()}>Counterfactual</button>
                <div className='flex-column'>
                  <span className='required'>Required:</span>
                  <span>{treatmentVars.length === 1 ? (<img src='/images/check.svg' />) :  (<img src='/images/close-2.svg' />)} One treatment variable</span>
                  <span>{counterfactualConditionsMet() ? (<img src='/images/check.svg' />) :  (<img src='/images/close-2.svg' />)} All other variables condition variables</span>
                </div>
              </div>
            </div>
            <div id="inference-results">
              {(estimatedAte || estimatedCate || estimatedCounterfactual) && (
                <>
                  <label htmlFor="" className="panel-label">Results</label>
                  <div className="panel">
                    <div id="results">
                      {estimatedAte && (
                        <div id="ate-results">
                          {loadingAte ? <>
                            <img className='inference-loading' src={"/images/loader.gif"} alt="" />
                          </> : <>
                            <p>Estimated ATE: {typeof estAte == 'number' ? parseFloat(estAte.toFixed(2)) : estAte }</p>
                            <p>True ATE: {typeof trueAte === 'number' ?  parseFloat(trueAte).toFixed(2) : trueAte}</p>
                          </>}
                          
                        </div>
                      )}
                      {estimatedCate && (
                        <div id="cate-results">
                          {loadingCate ? <>
                            <img className='inference-loading' src={"/images/loader.gif"} alt="" />
                          </> : <>
                            <p>Estimated CATE: {typeof estCate == 'number' ? parseFloat(estCate).toFixed(2) : estCate}</p>
                            <p>True CATE: {typeof trueCate == 'number' ? parseFloat(trueCate).toFixed(2) : trueCate}</p>
                          </>}
                        </div>
                      )}
                      {estimatedCounterfactual && (
                        <div id="ate-results">
                          {loadingCounterfactual ? <>
                            <img className='inference-loading' src={"/images/loader.gif"} alt="" />
                          </> : <>
                            <p>Estimated Counterfactual: {typeof estCounterfactual == 'number' ? parseFloat(estCounterfactual.toFixed(2)) : estCounterfactual }</p>
                            <p>True Counterfactual: {typeof trueCounterfactual === 'number' ?  parseFloat(trueCounterfactual).toFixed(2) : trueCounterfactual}</p>
                          </>}
                          
                        </div>
                      )}
                    </div>
                  </div>
                </>
              )} 
            </div>
          </div>
          <div>
            <label htmlFor="" className='panel-label'>Specifications</label>
            <div className="panel">
              <div>
               <label htmlFor="">Target Variable</label>
               <select name="data_type" id="data_type" defaultValue={''} onChange={(e) => setTargetVar(e.target.value)}>
                 <option value="" disabled>Choose a variable</option>
                  {varNames.map((name,index) =>{
                    return <option value={name} key={index}>{name}</option>
                  })}
                </select>
                <label htmlFor="">Prediction Model</label>
                <select name="data_type" id="data_type" defaultValue={''} onChange={(e) => setPredictionModel(e.target.value)}>
                  <option value="" disabled>Choose a Prediction Model</option>
                  {!isDiscrete && (
                    <option value="Linear Regression">Linear Regression</option>

                  )}
                  <option value="MLP Regression">MLP Regression</option>
                </select>
              </div>
            </div>
            
          </div>
          <div className='variables-section'>
          {(targetVar && predictionModel) && (
              <button className="btn-variant inference-button" onClick={() => addedVariable ? setAddedVariable(false) : setAddedVariable(true)}>Add Treatment Variable</button>
            )}
            <div className={addedVariable ? "panel inference-panel" : ''}>
              {addedVariable && (
                <div>
                  <label htmlFor="">Variable Name</label>
                  <select name="" id="" defaultValue={''}  onChange={(e) => setTreatmentVar(e.target.value)}>
                    <option value="" disabled>Select a Treatment Var</option>
                  {varNames.map((name,index) =>{
                    return <option value={name} key={index}>{name}</option>
                  })}
                  </select>
                  {treatmentVar && (
                    <>
                      <label htmlFor="">Treatment Value </label>{isDiscrete && (<span>Max: {maxStateVals[treatmentVar]}</span> )}
                      {isDiscrete 
                      ?
                        <input type="text" value={treatmentValue} onChange={(e) => e.target.value <= (maxStateVals[treatmentVar] || e.nativeEvent.inputType == 'deleteContentBackward') && e.nativeEvent.data != '.' ? setTreatmentValue(e.target.value) : e.preventDefault()} />
                      : 
                        <input type="text" value={treatmentValue} onChange={(e) => e.target.value < 9999 || e.nativeEvent.inputType == 'deleteContentBackward' || (e.nativeEvent.data == '-' && e.target.value.length < 2) ? setTreatmentValue(e.target.value) : e.preventDefault()} />
                      }
                      <label htmlFor="">Control Value </label> {isDiscrete && (<span>Max: {maxStateVals[treatmentVar]}</span> )}
                      {isDiscrete 
                      ?
                        <input type="text" value={controlValue} onChange={(e) => e.target.value <= (maxStateVals[treatmentVar] || e.nativeEvent.inputType == 'deleteContentBackward') && e.nativeEvent.data != '.' ? setControlValue(e.target.value) : e.preventDefault()} />
                      : 
                        <input type="text" value={controlValue} onChange={(e) => e.target.value < 9999 || e.nativeEvent.inputType == 'deleteContentBackward' || (e.nativeEvent.data == '-' && e.target.value.length < 2) ? setControlValue(e.target.value) : e.preventDefault()} />
                      }
                      <button className='btn inference-btn' onClick={() => handleAddTreatmentVars()}>Add</button>
                    </>
                  )}
                  <p className='error-message'>{treatmentError}</p>
                </div>
              )}
            </div>
            {treatmentVars.length != 0 && (
              <label htmlFor="" className='panel-label'>Treatment Variables</label>
            )}
            {(treatmentVars.length != 0) && (
              <div id={'treatment-vars-section'} className="panel">
                <table>
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Treatment</th>
                      <th>Control</th>
                    </tr>
                    <tr className='spacer'></tr>
                  </thead>
                  <tbody>
                    {treatmentVars.map((item, index) => {
                      return(
                        <>
                        <tr className='data-row' key={index}>
                          <td>{item[0]}</td>
                          <td>{item[1]}</td>
                          <td>{item[2]}</td>
                          <div className="button-cell"><a className='' onClick={() => deleteTreatmentVar(index)}>Remove</a></div>
                        </tr>
                        <tr className='spacer'></tr>
                        </>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
            
          </div>
          <div className="variables-section">
          {(targetVar && predictionModel) && (
              <button className="btn-variant inference-button" onClick={() => addedConditionVariable ? setAddedConditionVariable(false) : setAddedConditionVariable(true)}>Add Condition Variable</button>
            )}
            <div className={addedConditionVariable ? "panel inference-panel" : ''}>
              {addedConditionVariable && (
                <div>
                  <label htmlFor="">Variable Name</label>
                  <select name="" id="" defaultValue={''}  onChange={(e) => setConditionVar(e.target.value)}>
                    <option value="" disabled>Select a Condition Var</option>
                    {varNames.map((name,index) =>{
                      return <option value={name} key={index}>{name}</option>
                    })} 
                  </select>
                  {conditionVar && (
                    <>
                      <label htmlFor="">Condition Value </label>{isDiscrete && (<span>Max: {maxStateVals[conditionVar]}</span> )}
                      {isDiscrete 
                      ?
                        <input type="text" value={conditionValue} onChange={(e) => e.target.value <= (maxStateVals[conditionVar] || e.nativeEvent.inputType == 'deleteContentBackward') && e.nativeEvent.data != '.' ? setConditionValue(e.target.value) : e.preventDefault()} />
                      : 
                        <input type="text" value={conditionValue} onChange={(e) => e.target.value < 9999 || e.nativeEvent.inputType == 'deleteContentBackward' || (e.nativeEvent.data == '-' && e.target.value.length < 2)  ? setConditionValue(e.target.value) : e.preventDefault()} />
                      }
                      <button className='btn inference-btn' onClick={() => handleAddConditionVars()}>Add</button>
                    </>
                  )}
                  <p className='error-message'>{conditionError}</p>
                </div>
              )}
            </div>
            {conditionVars.length != 0 && (
              <label htmlFor="" className='panel-label' id='condition-label'>Condition Variables</label>
            )}
            {(conditionVars.length != 0) && (
              <div id={'condition-vars-section'} className="panel inference-panel">
                <table>
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Value</th>
                    </tr>
                    <tr className='spacer'></tr>
                  </thead>
                  <tbody>
                    {conditionVars.map((item, index) => {
                      return(
                        <>
                        <tr className='data-row' key={index}>
                          <td>{item[0]}</td>
                          <td>{item[1]}</td>
                          <div className="button-cell"><a className='' onClick={() => deleteConditionVar(index)}>Remove</a></div>
                        </tr>
                        <tr className='spacer'></tr>
                        </>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
    
  )
}

export default PerformCausalInference