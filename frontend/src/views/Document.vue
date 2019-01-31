<template>
    <v-container fluid pa-3>
        <v-layout row wrap>
            <v-flex md8 pa-2>
                <DocumentContent class="elevation-2"/>
            </v-flex>
            <v-flex md4 pa-2>
                <DocumentProperties/>
            </v-flex>
        </v-layout>
    </v-container>
</template>

<script>
import { mapActions } from 'vuex';
import DocumentContent from '../components/DocumentContent.vue';
import DocumentProperties from '../components/DocumentProperties.vue';

export default {
    components: {
        DocumentContent,
        DocumentProperties,
    },
    created() {
        this.fetchDocument(this.id);
    },
    watch: {
        $route() {
            this.fetchDocument(this.id);
        },
    },
    methods: {
        ...mapActions('document', [
            'fetchDocument',
        ]),
    },
    props: {
        id: String,
    },
};
</script>

<style>

</style>
