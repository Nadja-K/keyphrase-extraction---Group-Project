/* eslint-disable no-param-reassign */
import Vue from 'vue';
import Vuex from 'vuex';

import axios from 'axios';

Vue.use(Vuex);

export default new Vuex.Store({
    strict: true,

    state: {
        documents: {},
        document: {
            runs: [
                {
                    title: [],
                    lead: [],
                    text: [],
                },
            ],
        },
        dataset: 'heise',
        page: 0,
        showCandidateKeyphrases: true,
        showSelectedKeyphrases: true,
    },
    mutations: {
        setDocuments(state, documents) {
            state.documents = documents;
        },
        setDocument(state, document) {
            state.document = document;
        },
        setShowCandidateKeyphrases(state, value) {
            state.showCandidateKeyphrases = value;
        },
        setShowSelectedKeyphrases(state, value) {
            state.showSelectedKeyphrases = value;
        },
    },
    actions: {
        fetchDocuments({ commit }) {
            axios.get('/heise/documents').then((response) => {
                commit('setDocuments', response.data);
            });
        },
        fetchDocument({ commit }, documentId) {
            axios.get(`${this.dataset}/documents/${documentId}`).then((response) => {
                commit('setDocument', response.data);
            });
        },
        search({ commit }, query) {
            axios.get(`${this.dataset}/documents?q=${query}&page=8000`).then((response) => {
                commit('setDocuments', response.data);
            });
        },
        showCandidateKeyphrases({ commit }, value) {
            commit('setShowCandidateKeyphrases', value);
        },
        showSelectedKeyphrases({ commit }, value) {
            commit('setShowSelectedKeyphrases', value);
        },
    },
});
