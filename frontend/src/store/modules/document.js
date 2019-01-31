/* eslint-disable no-param-reassign */
import axios from 'axios';

export default {
    namespaced: true,

    state: {
        document: {
            runs: [
                {
                    title: [],
                    lead: [],
                    text: [],
                    num_clusters: 0,
                    settings: {
                        model: '',
                    },
                },
            ],
        },
    },
    getters: {
        selectedRun(state) {
            return state.document.runs[state.ui.selectedRunIndex] || {
                title: [],
                lead: [],
                text: [],
                settings: {
                    model: '',
                },
            };
        },
    },
    mutations: {
        setDocument(state, document) {
            state.document = document;
        },
    },
    actions: {
        fetchDocument({ commit }, documentId) {
            axios.get(`${this.dataset}/documents/${documentId}`).then((response) => {
                commit('setDocument', response.data);
            });
        },
    },
    modules: {
        ui: {
            namespaced: true,

            state: {
                commonProperties: {
                    showCandidateKeyphrases: true,
                    showSelectedKeyphrases: true,
                },
                keyClusterProperties: {
                    selectedClusters: [],
                },
                selectedRunIndex: 0,
                expandedPanels: [0,1,2],
            },
            mutations: {
                setCommonProperties(state, commonProperties) {
                    state.commonProperties = commonProperties;
                },
                setKeyClusterProperties(state, keyClusterProperties) {
                    state.keyClusterProperties = keyClusterProperties;
                },
                setSelectedRunIndex(state, selectedRunIndex) {
                    state.selectedRunIndex = selectedRunIndex;
                },
                setExpandedPanels(state, expandedPanels) {
                    state.expandedPanels = expandedPanels;
                },
            },
            actions: {
                setCommonProperty({ commit, state }, newValue) {
                    commit('setCommonProperties', { ...state.commonProperties, ...newValue });
                },
                setKeyClusterProperty({ commit, state }, newValue) {
                    commit('setKeyClusterProperties', { ...state.keyClusterProperties, ...newValue });
                },
                setSelectedRunIndex({ commit }, selectedRunIndex) {
                    commit('setSelectedRunIndex', selectedRunIndex);
                },
                setPanelExpanded({ commit, state }, { expanded, panel }) {
                    if (expanded) {
                        commit('setExpandedPanels', [...state.expandedPanels, panel]);
                    } else {
                        commit('setExpandedPanels', state.expandedPanels.filter(entry => entry !== panel));
                    }
                },
            },
        },
    },
};
