/* eslint-disable no-param-reassign */
import axios from 'axios';

export default {
    namespaced: true,

    state: {
        documents: {},
        dataset: 'heise',
        page: 0,
    },
    mutations: {
        setDocuments(state, documents) {
            state.documents = documents;
        },
    },
    actions: {
        fetchDocuments({ commit }) {
            axios.get('/heise/documents').then((response) => {
                commit('setDocuments', response.data);
            });
        },
        search({ commit }, query) {
            axios.get(`${this.dataset}/documents?q=${query}&page=8000`).then((response) => {
                commit('setDocuments', response.data);
            });
        },
    },
};
