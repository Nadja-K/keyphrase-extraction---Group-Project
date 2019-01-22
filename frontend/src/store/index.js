import Vue from 'vue';
import Vuex from 'vuex';

import documents from './modules/documents';
import document from './modules/document';

Vue.use(Vuex);

const debug = process.env.NODE_ENV !== 'production';

export default new Vuex.Store({
    modules: {
        documents,
        document,
    },
    strict: debug,
});
