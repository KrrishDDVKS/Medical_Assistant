import streamlit as st
import pandas as pd

st.markdown("<h1 style='text-align: center; color: black;'>MediConnect AI 🏥</h1>", unsafe_allow_html=True)
st.header('Patient Conversation Summaries:male-doctor:')
st.write(
        '''<b>Objective:</b> Provide doctors with a summarized view of previous patient conversations.<br>
        <b>Details:</b> Display key points, symptoms discussed, treatments suggested, and any follow-up actions from previous consultations.<br>
	<b>Benefit:</b>Helps doctors quickly understand the patient’s history, leading to more informed decisions and personalized care.
</span>''',
        unsafe_allow_html=True,)

if "summary" not in st.session_state:
    st.session_state["summary"] = ""

a=[' ', 'CAR0001', 'CAR0002', 'CAR0003', 'CAR0004', 'CAR0005', 'DER0001', 'GAS0001', 'GAS0002', 'GAS0003', 'GAS0004', 'GAS0005', 'GAS0007', 'GEN0001', 'MSK0001', 'MSK0003', 'MSK0004', 'MSK0005', 'MSK0006', 'MSK0007', 'MSK0008', 'MSK0009', 'MSK0010', 'MSK0011', 'MSK0012', 'MSK0013', 'MSK0014', 'MSK0015', 'MSK0016', 'MSK0017', 'MSK0018', 'MSK0019', 'MSK0020', 'MSK0021', 'MSK0022', 'MSK0023', 'MSK0024', 'MSK0025', 'MSK0026', 'MSK0027', 'MSK0028', 'MSK0029', 'MSK0031', 'MSK0032', 'MSK0033', 'MSK0034', 'MSK0035', 'MSK0036', 'MSK0037', 'MSK0038', 'MSK0039', 'MSK0040', 'MSK0041', 'MSK0042', 'MSK0043', 'MSK0044', 'MSK0045', 'MSK0046', 'MSK0049', 'MSK0050', 'RES0001', 'RES0002', 'RES0003', 'RES0004', 'RES0005', 'RES0006', 'RES0007', 'RES0008', 'RES0009', 'RES0010', 'RES0011', 'RES0012', 'RES0013', 'RES0014', 'RES0015', 'RES0016', 'RES0017', 'RES0018', 'RES0019', 'RES0020', 'RES0021', 'RES0022', 'RES0023', 'RES0024', 'RES0025', 'RES0026', 'RES0027', 'RES0028', 'RES0029', 'RES0030', 'RES0031', 'RES0032', 'RES0033', 'RES0034', 'RES0035', 'RES0036', 'RES0037', 'RES0038', 'RES0039', 'RES0042', 'RES0043', 'RES0044', 'RES0045', 'RES0046', 'RES0047', 'RES0048', 'RES0049', 'RES0050', 'RES0051', 'RES0052', 'RES0053', 'RES0054', 'RES0055', 'RES0056', 'RES0057', 'RES0058', 'RES0059', 'RES0060', 'RES0061', 'RES0062', 'RES0063', 'RES0064', 'RES0065', 'RES0066', 'RES0067', 'RES0068', 'RES0069', 'RES0070', 'RES0071', 'RES0072', 'RES0073', 'RES0074', 'RES0075', 'RES0076', 'RES0077', 'RES0078', 'RES0079', 'RES0080', 'RES0081', 'RES0082', 'RES0083', 'RES0084', 'RES0085', 'RES0086', 'RES0087', 'RES0088', 'RES0089', 'RES0090', 'RES0091', 'RES0092', 'RES0093', 'RES0094', 'RES0095', 'RES0096', 'RES0097', 'RES0098', 'RES0099', 'RES0100', 'RES0101', 'RES0102', 'RES0103', 'RES0104', 'RES0105', 'RES0106', 'RES0107', 'RES0108', 'RES0109', 'RES0110', 'RES0111', 'RES0112', 'RES0113', 'RES0114', 'RES0116', 'RES0117', 'RES0118', 'RES0119', 'RES0120', 'RES0121', 'RES0122', 'RES0123', 'RES0124', 'RES0125', 'RES0126', 'RES0127', 'RES0128', 'RES0129', 'RES0130', 'RES0131', 'RES0132', 'RES0133', 'RES0134', 'RES0135', 'RES0136', 'RES0137', 'RES0138', 'RES0139', 'RES0140', 'RES0141', 'RES0142', 'RES0143', 'RES0144', 'RES0145', 'RES0146', 'RES0147', 'RES0148', 'RES0149', 'RES0150', 'RES0151', 'RES0152', 'RES0153', 'RES0154', 'RES0155', 'RES0156', 'RES0158', 'RES0159', 'RES0160', 'RES0161', 'RES0162', 'RES0163', 'RES0164', 'RES0165', 'RES0166', 'RES0167', 'RES0168', 'RES0169', 'RES0170', 'RES0171', 'RES0172', 'RES0173', 'RES0174', 'RES0175', 'RES0176', 'RES0177', 'RES0178', 'RES0179', 'RES0180', 'RES0181', 'RES0182', 'RES0183', 'RES0184', 'RES0185', 'RES0186', 'RES0187', 'RES0188', 'RES0189', 'RES0190', 'RES0191', 'RES0192', 'RES0193', 'RES0194', 'RES0195', 'RES0196', 'RES0197', 'RES0198', 'RES0199', 'RES0200', 'RES0201', 'RES0202', 'RES0203', 'RES0204', 'RES0205', 'RES0206', 'RES0207', 'RES0208', 'RES0209', 'RES0210', 'RES0211', 'RES0212', 'RES0213', 'RES0214', 'RES0215', 'RES0216', 'RES0217']
if conid := st.selectbox(
    "Conversation ID",
    a,):
    if conid !=' ':
    
        df=pd.read_csv('./summaries.csv')
        st.session_state.summary=df.loc[df['File Name']==conid]['Summary'].values[0]
    else:
            st.session_state.summary=''

st.write(st.session_state["summary"])
