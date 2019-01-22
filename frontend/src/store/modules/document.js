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
                },
            ],
        },
        selectedRunIndex: 0,
    },
    getters: {
        selectedRun(state) {
            return state.document.runs[state.selectedRunIndex] || {
                title: [],
                lead: [],
                text: [],
                num_clusters: 0,
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
            },
            mutations: {
                setCommonProperties(state, commonProperties) {
                    state.commonProperties = commonProperties;
                },
                setKeyClusterProperties(state, keyClusterProperties) {
                    state.keyClusterProperties = keyClusterProperties;
                },
            },
            actions: {
                setCommonProperty({ commit, state }, newValue) {
                    commit('setCommonProperties', { ...state.commonProperties, ...newValue });
                },
                setKeyClusterProperty({ commit, state }, newValue) {
                    commit('setKeyClusterProperties', { ...state.keyClusterProperties, ...newValue });
                },
            },
        },
    },
};
