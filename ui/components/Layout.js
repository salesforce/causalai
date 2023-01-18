/**
 * This is the main component for setting up page layout
 * @param props
 * @returns {JSX.Element}
 * @constructor
 */
const Layout = props => {
  let style = props.clean ? {padding:0, margin:0, position:'relative', top:'-50px'}: {};
  return (
      <div className={"body-container flex-container-col"}>
        <nav>
          <div className={"left"}>
            <div><h1>Salesforce Causal AI Library</h1><div className={"sub-title"}>Demo</div></div>
          </div>
            <div className={"right"}>
                <a href="/" className="logo"> <img src="/images/Salesforce_Corporate_Logo_RGB.png" width="100" height="68"/></a>
            </div>
        </nav>
          <div className={"main-content"} style={style}>{props.children}</div>
          <footer>© Copyright 2021 Salesforce.com, inc.&nbsp;All rights reserved. Various trademarks held by their respective owners. Salesforce.com, inc. Salesforce Tower, 415 Mission Street, 3rd Floor, San Francisco, CA 94105, United States</footer>
      </div>
  )
}
export default Layout