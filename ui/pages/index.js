import { useState, useEffect } from "react";
import Layout from "../components/Layout";
import PerformCausalDiscovery from "../components/PerformCausalDiscovery";
import PerformCausalInference from "../components/PerformCausalInference";
import NodeGraph from "../components/NodeGraph";
import axios from 'axios';

const CausalAI = () => {
  const [activePage, setActivePage] = useState(0);
  const [activeTab, setActiveTab] = useState(0);
  const [dataGenerated, setDataGenerated] = useState(false);
  const [fileUploaded, setFileUploaded] = useState(false);

  const [dataType, setDataType] = useState('');
  const [isDiscrete, setIsDiscrete] = useState(null);
  const [discreteErrorMessage, setDiscreteErrorMessage] = useState('')
  const [numVars, setNumVars] = useState(3);
  const [maxLag, setMaxLag] = useState(null);
  const [numSamples, setNumSamples] = useState(1000);
  const [randomSeed, setRandomSeed] = useState(1);

  const [data, setData] = useState([])
  const [causalGraph, setCausalGraph] = useState({})
  const [varNames, setVarNames] = useState([])
  const [undirectedEdges, setUndirectedEdges] = useState([])
  const [graphUploaded, setGraphUploaded] = useState(false);
  
  const [downloadType, setDownloadType] = useState('')
  const [badGraphMessage, setBadGraphMessage] = useState(false)
  const [csv, setCsv] = useState('')
  const [badgraphUpload, setBadGraphUpload] = useState('')
  // const [undirectedEdgesError, setUndirectedEdgesError] = useState('')
  let undirectedEdgesError

  let fileReader;
  
  const handleSetTimeSeries = () => {
    setDataType('Time Series');
    setMaxLag(2)
  }
  const handleSetTabular = () => {
    setDataType('Tabular');
    setMaxLag(null)
  }

  const handleJSONFileRead = (e) => {
    setDiscreteErrorMessage('')
    let content = fileReader.result;
    content = JSON.parse(content)

    if(!Number.isInteger(content['data'][1][1]) && isDiscrete){
      setDiscreteErrorMessage('The selected data type is discrete but the uploaded data is continuous')
    }
    else if(Number.isInteger(content['data'][1][1]) && !isDiscrete){
      setDiscreteErrorMessage('The selected data type is continuous but the uploaded data is discrete')
    }
    else if(content['varNames'].length > 40){
      setDiscreteErrorMessage('This file has too many variables. Please upload a file with 40 variables or less')
    }
    else{
      setData(content['data'])
      setVarNames(content['varNames'])
      setFileUploaded(true)
      setDataGenerated(false)
      setCausalGraph({})
      setActiveTab(1);
    }

  };
  const handleCSVFileRead = (e) => {
    setDiscreteErrorMessage('')
    let varNames = []
    let data = []
    let content = fileReader.result;
    varNames = content.slice(0, content.indexOf("\n")).split(',');
    let rows = content.slice(content.indexOf("\n") + 1).split("\n");
    // loops through the csv file and converts all strings to numbers
    let tempFile =[]
    for(let arr in rows){
      let temp = Array(rows[arr].split(','))
      for(let i in temp){
        let tempArr = []
        for(let j in temp[i]){
          tempArr.push(Number(temp[i][j]))
        }
        tempFile.push(tempArr)
      }
    }
    rows = tempFile

    for(let arr in rows){
      data.push(rows[arr])
    }
    if(!Number.isInteger(data[1][1]) && isDiscrete){
      setDiscreteErrorMessage('The selected data type is discrete but the uploaded data is continuous')
    }
    else if(Number.isInteger(data[1][1]) && !isDiscrete){
      setDiscreteErrorMessage('The selected data type is continuous but the uploaded data is discrete')
    }
    else if(varNames > 40){
      setDiscreteErrorMessage('This file has too many variables. Please upload a file with 40 variables or less')
    }
    else{
      setData(data)
      setVarNames(varNames)
      setFileUploaded(true)
      setCausalGraph({})
      setDataGenerated(false)
      setActiveTab(1)
      setDiscreteErrorMessage('')
    }
  };
  
  const handleFileChosen = (file) => {
    if(file.size > 9000000){
      setDiscreteErrorMessage('The uploaded file too large. Please upload a file that is less than 9MB')
      return;
    }
    setDataGenerated(false)
    fileReader = new FileReader();
    if(file.type === 'application/json'){
      fileReader.onloadend = handleJSONFileRead;
    }
    else{
      fileReader.onloadend = handleCSVFileRead;
    }
    fileReader.readAsText(file);
  };


  const checkVariables = (graph) => {
    let keyArray = [];
    for (let [key, value] of Object.entries(graph)) {
      keyArray.push(key)
    }

    if(keyArray.length != varNames.length){
      return false;
    }
    else{
      for(let i = 0; i < keyArray.length; i++ ){
        if(keyArray[i] != varNames[i]){
          return false;
        }
      }
      return true;
    }
    return true;
  }
  const buildCSV =  async () =>{
    let tempCsv = ''
    tempCsv =  tempCsv.concat(varNames + '\n')
    for(let i = 0; i < data.length; i++){
      while(i < data.length-1){
        tempCsv = tempCsv.concat(data[i] + '\n')
        i++;  
      }
      tempCsv = tempCsv.concat(data[i]);
    }
    setCsv(tempCsv)
  }
  const handleCSV = () =>{ 
    setDownloadType('csv');
    buildCSV();
  }
  const handleGraphUpload = (file) => {
    fileReader = new FileReader();
    fileReader.onloadend = handleGraphFileRead;
    fileReader.readAsText(file);

  };

  const handleGraphFileRead = async (e) => {
    let timeSeriesCheck;
    setBadGraphMessage('')
    let content = await fileReader.result;
    content = JSON.parse(content)
    await hasUndirectedEdges(content)
    for(let i = 0; i < Object.keys(content).length; i++){
      if( content[Object.keys(content)[i]][0] != undefined && content[Object.keys(content)[i]][0][1] != undefined){
        timeSeriesCheck = true;
      }
      else{
        timeSeriesCheck = false;
      }
    }
    if(undirectedEdgesError){
      setBadGraphMessage(undirectedEdgesError);
      return;
    }
    if(dataType === 'Tabular' && timeSeriesCheck){
      setBadGraphMessage('Selected data type is tabular but uploaded graph is time series')
      return;
    } 
    else if(dataType === 'Time Series' && !timeSeriesCheck){
      setBadGraphMessage('Selected data type is time series but uploaded graph is tabular')
      return;
    }
    if(checkVariables(content)){
      await setCausalGraph(content)
      await getUndirectedEdges(content)
      setGraphUploaded('Uploaded graph must have same variable names as uploaded data')

    }
    else{
      setBadGraphMessage('Uploaded graph must have same variable names as uploaded data')
    }

  };
  
  const launchCausalInfernece = async () => {
    if(fileUploaded){
      setActivePage(3);

    }
    else{
      setActivePage(3);
    }
  }


  useEffect(() => {
    checkCausalGraph();
  },[causalGraph]);


  const hasUndirectedEdges = async (graph) => {

    let data = new FormData();
    let causalArray =  JSON.stringify(graph);

		data.append('graph', causalArray);

    var config = {
			method: 'post',
			url: 'http://127.0.0.1:5000/has_undirected_edges',
			headers: { 
				'Content-Type': 'application/json'
			},
			data : data
		}

    await axios(config)
      .then(function (response) {
        undirectedEdgesError = response.data.msg
      })
      .catch(function (error) {
        console.log(error);
      });
  }
  
  const getUndirectedEdges = (graph) => {

    let data = new FormData();
    let causalArray =  JSON.stringify(graph);

		data.append('causal_graph', causalArray);
    data.append('data_type', dataType)

    var config = {
			method: 'post',
			url: 'http://127.0.0.1:5000/undirected_edges',
			headers: { 
				'Content-Type': 'application/json'
			},
			data : data
		}

    axios(config)
      .then(function (response) {
        setUndirectedEdges(response.data.undirected_edges)

      })
      .catch(function (error) {
        console.log(error);
      });
  }

  const checkCausalGraph = () => {

    let data = new FormData();
    let causalArray =  JSON.stringify(causalGraph);

		data.append('graph', causalArray);

    var config = {
			method: 'post',
			url: 'http://127.0.0.1:5000/check_causal_graph_format',
			headers: { 
				'Content-Type': 'application/json'
			},
			data : data
		}

    axios(config)
      .then(function (response) {
        if(response.data.bool === false){
          setBadGraphUpload(response.data.msg)
          setGraphUploaded(false)
        }
      })
      .catch(function (error) {
        console.log(error);
      });

  }

  

  const handleGenerateData = () => {
    setDataGenerated(false);
    setGraphUploaded(false);
		let data = new FormData();
		data.append('data_type', dataType);
    data.append('isDiscrete', isDiscrete);
		data.append('num_vars', numVars);
		data.append('num_samples', numSamples);
		data.append('max_lag', maxLag);
    data.append('random_seed', randomSeed);

    var config = {
			method: 'post',
			url: 'http://127.0.0.1:5000/generate_data',
			headers: { 
				'Content-Type': 'application/json'
			},
			data : data
		}

    axios(config)
      .then(function (response) {
        setData(response.data.data_array);
        setCausalGraph(response.data.causal_graph);
        setVarNames(response.data.var_names);
        setDataGenerated(true);
        setFileUploaded(false)
        // setActivePage(1);
      })
      .catch(function (error) {
        console.log(error);
      });
  }

  const resetGraphUpload = () =>{
    setGraphUploaded(false)
    setCausalGraph({})
    
  }
  return(
    <Layout currentMenu={"casual-ai"} title={'causal-ai'}>
      {activePage === 0 && (
        <div id="input-page">

          <div id="input-page-inputs">
            <label htmlFor="" className="panel-label">Add Data</label>
            <div className="panel">
              <label htmlFor="">Data Type</label>
              <div className="radio-buttons">
                <div className="radio-button" id={'first-radio'}>
                  <div>
                    <input type="radio" id='tabular' name="data_type" value={'tabular'} onClick={() => handleSetTabular()} defaultChecked={dataType === 'Tabular'}  />
                    <label htmlFor="tabular">Tabular</label>
                  </div>
                  <div>
                    <input type="radio" id='time series' name="data_type" value={'time series'} onClick={() => handleSetTimeSeries()} defaultChecked={dataType === 'Time Series'}  />
                    <label htmlFor="time series">Time Series</label>
                  </div>
                </div>
                <hr />
                <div className="radio-button" id='second-radio'>
                  <div>
                    <input type="radio" id='discrete' name="isDiscrete" value={'discrete'} onClick={() => setIsDiscrete(true)} defaultChecked={isDiscrete} />
                    <label htmlFor="discrete">Discrete</label>
                  </div>
                  <div>
                    <input type="radio" id='continuous' name="isDiscrete" value={'continuous'} onClick={() => setIsDiscrete(false)}  defaultChecked={isDiscrete === false}/>
                    <label htmlFor="continuous">Continuous</label>
                  </div>
                </div>
              </div>

              </div>
              {(dataType && (isDiscrete != null)) && (
                <div className="panel">
              <div>
              {discreteErrorMessage && (
                    <p className="example-file error-message">{discreteErrorMessage}</p>
                  )}
                <div className="buttons">

                  {!fileUploaded && (
                  <div id='file-upload' aria-disabled="false">
                    <div>
                      <img src="/images/upload.svg" />
                      <div id="select-a-file">
                        <p className="select-a-file-text">Select a file</p>
                        <p>or drag it here</p>
                      </div>
                    </div>
                    <input 
                      type="file" 
                      accept='.json, .csv' 
                      onChange={(e) => handleFileChosen(e.target.files[0])}
                    />
                  </div>
                  )}
                  {fileUploaded && (
                    <>
                      <div id="file-upload">
                          <img src="/images/check.svg" alt="" />

                        <div id="select-a-file">
                          <p className="file-selected-text">File Uploaded!</p>
                        </div>
                      </div>
                      <p className="example-file">Want to upload a different file? Click <a type="button" className="different-file" onClick={() => setFileUploaded(false)}>here</a></p>
                    </>
                  )}
                  {!fileUploaded && (
                  <p className="example-file">Need an example? Download sample files <a type="button" href="/uploads/sample_files_data.zip">here</a></p>
                  )}

                  <p>OR</p>
                  <button className='btn-variant' onClick={() => activeTab === 2 ?  setActiveTab(1) : setActiveTab(2)}>Generate Data</button>
             

                </div>

                {activeTab === 2 && (
                  <div className="inputs">
                    <div className="input">
                      <label>Number of Variables:</label>
                      <select defaultValue={numVars} onChange={(e) => setNumVars(e.target.value)}>
                        <option value={3}>3</option>
                        <option value={5}>5</option>
                      </select>
                    </div>
                    {dataType === 'Time Series' && (
                      <div className="input">
                        <label>Max Lag:</label>
                        <select defaultValue={maxLag} onChange={(e) => setMaxLag(e.target.value)}>
                          <option value={2}>2</option>
                          <option value={4}>4</option>
                          <option>6</option>
                        </select>
                      </div>
                    )}

                    <div className="input">
                      <label>Number of Samples:</label>
                      <select defaultValue={numSamples} onChange={(e) => setNumSamples(e.target.value)}>
                        <option value={1000}>1000</option>
                        <option value={3000}>3000</option>
                        <option value={5000}>5000</option>
                        <option value={7000}>7000</option>
                      </select>
                    </div>
                    <div className="input">
                      <label>Random Seed:</label>
                      <select defaultValue={randomSeed} onChange={(e) => setRandomSeed(e.target.value)}>
                        <option value={1}>1</option>
                        <option value={2}>2</option>
                        <option value={3}>3</option>
                        <option value={4}>4</option>
                        <option value={5}>5</option>
                        <option value={6}>6</option>
                        <option value={7}>7</option>
                        <option value={8}>8</option>
                        <option value={9}>9</option>
                        <option value={10}>10</option>
                      </select>
                    </div>
                    <button className='btn' onClick={() => handleGenerateData()}>Generate</button>
                  </div>
                )}


              </div>
            </div>
            )}
            {dataGenerated && (
              <>
                <label htmlFor="" className="panel-label">Download Data</label>
                <div className="panel">
                  <div className="download-section">
                    <div className="radio-button" id={'download-radio'}>
                      <div>
                        <input type="radio" id='json' name="download_type" value={'json'} onClick={() => setDownloadType('json')} />
                        <label htmlFor="json">JSON</label>
                      </div>
                      <div>
                        <input type="radio" id='csv' name="download_type" value={'csv'} onClick={() => handleCSV()} />
                        <label htmlFor="csv">CSV</label>
                      </div>
                    </div>
                  </div>
                  {downloadType === 'json' && (
                    <a 
                      type='button' 
                      id='download-graph-button' 
                      className='btn'
                      href={`data:text/json;charset=utf-8,${encodeURIComponent(
                        JSON.stringify({'data': data, 'varNames': varNames})
                      )}`}
                      download="data.json"
                    >
                      Download Data
                    </a>
                  )}
                  {downloadType === 'csv' && (
                    <a 
                      type='button' 
                      id='download-graph-button' 
                      className='btn'
                      href={`data:text/csv;charset=utf-8,${encodeURIComponent(
                        csv
                      )}`}
                      download="data.csv"
                    >
                      Download Data

                    </a>
                  )}
                </div>
                <div>
                <a 
                  type='button' 
                  id='download-graph-button' 
                  className='btn'
                  href={`data:text/json;charset=utf-8,${encodeURIComponent(
                    JSON.stringify(causalGraph)
                  )}`}
                  download="causal-graph.json"
                >
                  Download Graph

                </a>
                </div>
              </>
            )}
            {(fileUploaded && !graphUploaded) && (
              <>
                <label htmlFor="" className="panel-label">Upload Causal Graph</label>
                <p id="graph-disclaimer">Causal Graph is required for Causal Inference</p>

                {badgraphUpload && (
                  <>
                    <br />
                    <p id="bad-graph">{badgraphUpload}</p>
                  </>
                )}
                
                {badGraphMessage && (
                  <>
                    <br />
                    <p id="bad-graph">{badGraphMessage}</p>
                  </>
                )}
                <div className="panel">
                  <div id='file-upload' aria-disabled="false">
                  <div>
                    <img src="/images/upload.svg" />
                    <div id="select-a-file">
                      <p className="select-a-file-text">Select a file</p>
                      <p>or drag it here</p>
                    </div>
                  </div>
                  <input 
                    type="file" 
                    accept='.json' 
                    onChange={(e) => handleGraphUpload(e.target.files[0])}
                  />
                </div>
                {!graphUploaded && (
                  <p className="example-file">Need an example? Download sample files <a type="button" href="/uploads/sample_files_graph.zip">here</a></p>
                )}
              </div>
            </>
            )}
            {(fileUploaded && !graphUploaded) && (
              <button className="btn" id='solo-discovery-btn' onClick={() => setActivePage(2)}>Perform Causal Discovery</button>
            )}
          
            {graphUploaded && (
              <>
                <label htmlFor="" className="panel-label">Upload Causal Graph</label>
                <div className="panel">
                  <div className="buttons">
                    <div id="file-upload">
                      <img src="/images/check.svg" alt="" />

                      <div id="select-a-file">
                        <p className="file-selected-text">File Uploaded!</p>
                      </div>
                    </div>
                    <p className="example-file">Want to upload a different file? Click <a type="button" className="different-file" onClick={() => resetGraphUpload()}>here</a></p>

                  </div>
                </div>

              </>
            )}
          </div>
          <div id="input-page-graph">
            {((dataGenerated || graphUploaded)) && (
              <div>
                <label htmlFor="" className="panel-label">Ground Truth Graph</label>
                <NodeGraph data={causalGraph} dataType={dataType} graphId={'cy'} />

                <div className="perform-buttons">
                  <button className='btn' onClick={() => setActivePage(2)}>Perform Causual Discovery</button>
                  <button className="btn" onClick={() => launchCausalInfernece()}>Perform Causual Inference</button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      {activePage === 2 && (
        <PerformCausalDiscovery dataType={dataType} dataArray={data} varNames={varNames} causalGraph={causalGraph} isDiscrete={isDiscrete} setActivePage={setActivePage} />      
      )}
      {activePage === 3 && (
        <PerformCausalInference dataType={dataType} data={data} varNames={varNames} causalGraph={causalGraph} isDataGenerated={dataGenerated} maxLag={maxLag} randomSeed={randomSeed} numSamples={numSamples} numVars={numVars} isDiscrete={isDiscrete} fileUploaded={fileUploaded} setActivePage={setActivePage} undirectedEdges={undirectedEdges} /> 
      )}

    </Layout>
  )
}

export default CausalAI;