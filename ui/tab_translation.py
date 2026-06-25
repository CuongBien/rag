import streamlit as st
from services.translation_service import translate_molecule, render_molecule_image

def render_translation_tab():
    st.header("SMILES ↔ IUPAC Translation")
    st.markdown("Tier 1: Local Cache ➔ Tier 2: Local ML Model ➔ Tier 3: PubChem API")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        direction = st.radio("Translation Direction", ["SMILES to IUPAC", "IUPAC to SMILES"])
        query = st.text_input("Enter your query:", placeholder="e.g., CCO or Ethanol")
        translate_btn = st.button("Translate", type="primary")
        
    if translate_btn:
        if not query.strip():
            st.error("Please enter a query.")
        else:
            with st.spinner("Translating..."):
                dir_key = "smiles_to_iupac" if direction == "SMILES to IUPAC" else "iupac_to_smiles"
                result, source = translate_molecule(query.strip(), dir_key)
                
                if result:
                    st.success(f"**Result:** {result}")
                    st.info(f"**Source:** {source}")
                    
                    st.subheader("2D Structure")
                    smiles_for_image = query.strip() if direction == "SMILES to IUPAC" else result
                    render_molecule_image(smiles_for_image)
                else:
                    st.error("Translation failed. All 3 tiers (Cache, Model, API) could not resolve the query.")
